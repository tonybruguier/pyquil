##############################################################################
# Copyright 2016-2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
import logging
import sys
import warnings

import rpcq
from requests.exceptions import RequestException
from typing import Dict, Any, Optional, Union, cast, List, Tuple

from rpcq._base import Message, to_json, from_json
from rpcq.messages import (
    QuiltCalibrationsRequest,
    QuiltCalibrationsResponse,
    QuiltBinaryExecutableRequest,
    QuiltBinaryExecutableResponse,
    NativeQuilRequest,
    TargetDevice,
    PyQuilExecutableResponse,
    ParameterSpec,
)
from urllib.parse import urljoin

from pyquil import api
from pyquil.api._qac import AbstractCompiler
from pyquil.api._error_reporting import _record_call
from pyquil.api._errors import UserMessageError
from pyquil.api._rewrite_arithmetic import rewrite_arithmetic
from pyquil.device._main import AbstractDevice, Device
from pyquil.parser import parse_program
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Measurement, Declare
from pyquil.version import __version__

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass


_log = logging.getLogger(__name__)

PYQUIL_PROGRAM_PROPERTIES = ["native_quil_metadata", "num_shots"]


class QuilcVersionMismatch(Exception):
    pass


class QuilcNotRunning(Exception):
    pass


class QPUCompilerNotRunning(Exception):
    pass


def check_quilc_version(version_dict: Dict[str, str]) -> None:
    """
    Verify that there is no mismatch between pyquil and quilc versions.

    :param version_dict: Dictionary containing version information about quilc.
    """
    quilc_version = version_dict["quilc"]
    major, minor, patch = map(int, quilc_version.split("."))
    if major == 1 and minor < 8:
        raise QuilcVersionMismatch(
            "Must use quilc >= 1.8.0 with pyquil >= 2.8.0, but you "
            f"have quilc {quilc_version} and pyquil {__version__}"
        )


def parse_mref(val: str) -> MemoryReference:
    """ Parse a memory reference from its string representation. """
    val = val.strip()
    try:
        if val[-1] == "]":
            name, offset = val.split("[")
            return MemoryReference(name, int(offset[:-1]))
        else:
            return MemoryReference(val)
    except Exception:
        raise ValueError(f"Unable to parse memory reference {val}.")


def _extract_attribute_dictionary_from_program(program: Program) -> Dict[str, Any]:
    """
    Collects the attributes from PYQUIL_PROGRAM_PROPERTIES on the Program object program
    into a dictionary.

    :param program: Program to collect attributes from.
    :return: Dictionary of attributes, keyed on the string attribute name.
    """
    attrs = {}
    for prop in PYQUIL_PROGRAM_PROPERTIES:
        attrs[prop] = getattr(program, prop)
    return attrs


def _extract_program_from_pyquil_executable_response(response: PyQuilExecutableResponse) -> Program:
    """
    Unpacks a rpcq PyQuilExecutableResponse object into a pyQuil Program object.

    :param response: PyQuilExecutableResponse object to be unpacked.
    :return: Resulting pyQuil Program object.
    """
    p = Program(response.program)
    for attr, val in response.attributes.items():
        setattr(p, attr, val)
    return p


def _collect_memory_descriptors(program: Program) -> Dict[str, ParameterSpec]:
    """Collect Declare instructions that are important for building the patch table.

    This is secretly stored on BinaryExecutableResponse. We're careful to make sure
    these objects are json serializable.

    :return: A dictionary of variable names to specs about the declared region.
    """
    return {
        instr.name: ParameterSpec(type=instr.memory_type, length=instr.memory_size)
        for instr in program
        if isinstance(instr, Declare)
    }


# TODO This should be deleted once native_quil_to_executable no longer
# uses it.
def _collect_classical_memory_write_locations(
    program: Program,
) -> List[Optional[Tuple[MemoryReference, str]]]:
    """Collect classical memory locations that are the destination of MEASURE instructions
    These locations are important for munging output buffers returned from the QPU
    server to the shape expected by the user.
    This is secretly stored on BinaryExecutableResponse. We're careful to make sure
    these objects are json serializable.
    :return: list whose value `(q, m)` at index `addr` records that the `m`-th measurement of
        qubit `q` was measured into `ro` address `addr`. A value of `None` means nothing was
        measured into `ro` address `addr`.
    """
    ro_size = None
    for instr in program:
        if isinstance(instr, Declare) and instr.name == "ro":
            if ro_size is not None:
                raise ValueError(
                    "I found multiple places where a register named `ro` is declared! "
                    "Please only declare one register named `ro`."
                )
            ro_size = instr.memory_size

    ro_sources: Dict[int, Tuple[MemoryReference, str]] = {}

    for instr in program:
        if isinstance(instr, Measurement):
            q = instr.qubit.index
            if instr.classical_reg:
                offset = instr.classical_reg.offset
                assert instr.classical_reg.name == "ro", instr.classical_reg.name
                if offset in ro_sources:
                    _log.warning(
                        f"Overwriting the measured result in register "
                        f"{instr.classical_reg} from qubit {ro_sources[offset]} "
                        f"to qubit {q}"
                    )
                # we track how often each qubit is measured (per shot) and into which register it is
                # measured in its n-th measurement.
                ro_sources[offset] = (MemoryReference(name="ro", offset=offset), f"q{q}")
    if ro_size:
        return [ro_sources.get(i) for i in range(ro_size)]
    elif ro_sources:
        raise ValueError(
            "Found MEASURE instructions, but no 'ro' or 'ro_table' region was declared."
        )
    else:
        return []


class QPUCompiler(AbstractCompiler):
    @_record_call
    def __init__(
        self,
        device: AbstractDevice,
        quilc_endpoint: Optional[str] = None,
        qpu_compiler_endpoint: Optional[str] = None,
        timeout: float = 10,
        name: Optional[str] = None,
        client: Optional[api.Client] = None,
    ) -> None:
        """
        Client to communicate with the Compiler Server.

        :param quilc_endpoint: Optional TCP or IPC endpoint of the Quil Compiler (quilc). If omitted, endpoint will be
               derived from QCS client.
        :param qpu_compiler_endpoint: Optional TCP or IPC endpoint of the QPU Compiler. If omitted, endpoint will be
               derived from QCS client.
        :param device: PyQuil Device object to use as compilation target.
        :param timeout: Number of seconds to wait for a response from the client.
        :param name: Name of the lattice being targeted.
        :param client: Optional QCS client. If none is provided, a default client will be created.
        """
        self._api_client = client or api.Client()

        self.timeout = timeout

        quilc_endpoint = quilc_endpoint or client.quilc_url

        if not quilc_endpoint.startswith("tcp://"):
            # TODO(andrew): Update env var in message
            raise ValueError(
                f"PyQuil versions >= 2.4 can only talk to quilc "
                f"versions >= 1.4 over network RPCQ.  You've supplied the "
                f"endpoint '{quilc_endpoint}', but this doesn't look like a network "
                f"ZeroMQ address, which has the form 'tcp://domain:port'. "
                f"You might try clearing (or correcting) your COMPILER_URL "
                f"environment variable and removing (or correcting) the "
                f"compiler_server_address line from your .forest_config file."
            )

        self.quilc_client = rpcq.Client(quilc_endpoint, timeout=timeout)

        self.qpu_compiler_endpoint = qpu_compiler_endpoint
        self._qpu_compiler_client: Optional[Union[rpcq.Client, HTTPCompilerClient]] = None
        self._calibration_program = None

        self._device = device
        td = TargetDevice(isa=device.get_isa().to_dict(), specs=None)  # type: ignore
        self.target_device = td
        self.name = name

        try:
            self.connect()
        except QuilcNotRunning as e:
            warnings.warn(f"{e}. Compilation using quilc will not be available.")
        except QPUCompilerNotRunning as e:
            warnings.warn(f"{e}. Compilation using the QPU compiler will not be available.")

    @property
    def qpu_compiler_client(self) -> Optional[Union[rpcq.Client, "HTTPCompilerClient"]]:
        if not self._qpu_compiler_client:
            endpoint = self.qpu_compiler_endpoint or self._api_client.qpu_compiler_url

            if endpoint.startswith(("http://", "https://")):
                assert isinstance(self._device, Device)
                device_endpoint = urljoin(endpoint, f'devices/{self._device._raw["device_name"]}/')
                self._qpu_compiler_client = HTTPCompilerClient(
                    endpoint=device_endpoint, #session=self.session TODO(andrew): use client
                )
            elif endpoint.startswith("tcp://"):
                self._qpu_compiler_client = rpcq.Client(endpoint, timeout=self.timeout)
            else:
                raise UserMessageError(
                    "Invalid endpoint provided to QPUCompiler. Expected protocol in [http://, "
                    f"https://, tcp://], but received endpoint {endpoint}"
                )

        assert (
            isinstance(self._qpu_compiler_client, (rpcq.Client, HTTPCompilerClient))
            or self._qpu_compiler_client is None
        )
        return self._qpu_compiler_client

    def connect(self) -> None:
        self._connect_quilc()
        if self.qpu_compiler_client:
            self._connect_qpu_compiler()

    def _connect_quilc(self) -> None:
        try:
            quilc_version_dict = self.quilc_client.call("get_version_info")
            check_quilc_version(quilc_version_dict)
        except TimeoutError:
            raise QuilcNotRunning(
                f"Request to quilc at {self.quilc_client.endpoint} timed out. "
                "This could mean that quilc is not running, is not reachable, or is "
                "responding slowly."
            )

    def _connect_qpu_compiler(self) -> None:
        assert self.qpu_compiler_client is not None
        try:
            self.qpu_compiler_client.call("get_version_info")
        except TimeoutError:
            raise QPUCompilerNotRunning(
                f"Request to the QPU Compiler at {self.qpu_compiler_client.endpoint} "
                "timed out. "
                "This could mean that the service is not reachable or is responding slowly."
            )

    def get_version_info(self) -> Dict[str, Any]:
        quilc_version_info = self.quilc_client.call("get_version_info")
        if self.qpu_compiler_client:
            qpu_compiler_version_info = self.qpu_compiler_client.call("get_version_info")
            return {"quilc": quilc_version_info, "qpu_compiler": qpu_compiler_version_info}
        return {"quilc": quilc_version_info}

    @_record_call
    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool] = None) -> Program:
        self._connect_quilc()
        request = NativeQuilRequest(
            quil=program.out(calibrations=False), target_device=self.target_device
        )
        response = self.quilc_client.call(
            "quil_to_native_quil", request, protoquil=protoquil
        ).asdict()
        nq_program = parse_program(response["quil"])
        nq_program.native_quil_metadata = response["metadata"]
        nq_program.num_shots = program.num_shots
        nq_program._calibrations = program.calibrations
        return nq_program

    @_record_call
    def native_quil_to_executable(
        self, nq_program: Program, *, debug: bool = False
    ) -> Optional[QuiltBinaryExecutableResponse]:
        if not self.qpu_compiler_client:
            raise UserMessageError(
                "It looks like you're trying to compile to an executable, but "
                "do not have access to the QPU compiler endpoint. Make sure you "
                "are engaged to the QPU before trying to do this."
            )

        self._connect_qpu_compiler()

        arithmetic_response = rewrite_arithmetic(nq_program)

        request = QuiltBinaryExecutableRequest(
            quilt=arithmetic_response.quil, num_shots=nq_program.num_shots
        )
        response = cast(
            QuiltBinaryExecutableResponse,
            self.qpu_compiler_client.call(
                "native_quilt_to_binary", request, rpc_timeout=self.timeout
            ),
        )

        response.recalculation_table = arithmetic_response.recalculation_table  # type: ignore
        response.memory_descriptors = _collect_memory_descriptors(nq_program)

        # Convert strings to MemoryReference for downstream processing.
        response.ro_sources = [(parse_mref(mref), source) for mref, source in response.ro_sources]

        # TODO (kalzoo): this is a temporary workaround to migrate memory location parsing from
        # the client side (where it was pre-quilt) to the service side. In some cases, the service
        # won't return ro_sources, and so we can fall back to parsing the change on the client side.
        if response.ro_sources == []:
            response.ro_sources = _collect_classical_memory_write_locations(nq_program)

        if not debug:
            response.debug = {}

        return response

    @_record_call
    def get_calibration_program(self) -> Program:
        """
        Get the Quil-T calibration program associated with the underlying QPU.

        A calibration program contains a number of DEFCAL, DEFWAVEFORM, and
        DEFFRAME instructions. In sum, those instructions describe how a Quil-T
        program should be translated into analog instructions for execution on
        hardware.

        :returns: A Program object containing the calibration definitions."""
        self._connect_qpu_compiler()
        request = QuiltCalibrationsRequest(target_device=self.target_device)
        if not self.qpu_compiler_client:
            raise UserMessageError(
                # TODO(andrew): Update this
                # the pyquil config docs?
                "It looks like you're trying to request Quil-T calibrations, but "
                "do not have access to the QPU compiler endpoint. Make sure you "
                "are engaged to the QPU or have configured qpu_compiler_endpoint "
                "in your pyquil configuration."
            )
        response = cast(
            QuiltCalibrationsResponse,
            self.qpu_compiler_client.call("get_quilt_calibrations", request),
        )
        calibration_program = parse_program(response.quilt)
        return calibration_program

    @_record_call
    def refresh_calibration_program(self) -> None:
        """Refresh the calibration program cache."""
        self._calibration_program = self.get_calibration_program()

    @property
    def calibration_program(self) -> Program:
        """Cached calibrations."""
        if self._calibration_program is None:
            self.refresh_calibration_program()

        if self._calibration_program is None:
            raise RuntimeError("Could not refresh calibrations")
        else:
            return self._calibration_program

    @_record_call
    def reset(self) -> None:
        """
        Reset the state of the QPUCompiler Client connections.
        """
        self._qpu_compiler_client = None

    def set_timeout(self, timeout: float) -> None:
        """
        Set timeout for each individual stage of compilation.

        :param timeout: Timeout value for each compilation stage, in seconds. If the stage does not
            complete within this threshold, an exception is raised.
        """
        if timeout < 0:
            raise ValueError(f"Cannot set timeout to negative value {timeout}")

        self.timeout = timeout
        self.quilc_client.rpc_timeout = timeout


class QVMCompiler(AbstractCompiler):
    @_record_call
    def __init__(self, endpoint: str, device: AbstractDevice, timeout: float = 10) -> None:
        """
        Client to communicate with the Compiler Server.

        :param endpoint: TCP or IPC endpoint of the Compiler Server
        :param device: PyQuil Device object to use as compilation target
        :param timeout: Timeout value for each compilation stage, in seconds. If the stage does not
            complete within this threshold, an exception is raised.
        """

        if not endpoint.startswith("tcp://"):
            raise ValueError(
                f"PyQuil versions >= 2.4 can only talk to quilc "
                f"versions >= 1.4 over network RPCQ.  You've supplied the "
                f"endpoint '{endpoint}', but this doesn't look like a network "
                f"ZeroMQ address, which has the form 'tcp://domain:port'. "
                f"You might try clearing (or correcting) your COMPILER_URL "
                f"environment variable and removing (or correcting) the "
                f"compiler_server_address line from your .forest_config file."
            )

        self.endpoint = endpoint
        self.client = rpcq.Client(endpoint, timeout=timeout)
        td = TargetDevice(isa=device.get_isa().to_dict(), specs=None)  # type: ignore
        self.target_device = td

        try:
            self.connect()
        except QuilcNotRunning as e:
            warnings.warn(f"{e}. Compilation using quilc will not be available.")

    def connect(self) -> None:
        try:
            version_dict = self.get_version_info()
            check_quilc_version(version_dict)
        except TimeoutError:
            raise QuilcNotRunning(
                f"Request to quilc at {self.client.endpoint} timed out. "
                "This could mean that quilc is not running, is not reachable, or is "
                "responding slowly."
            )

    def get_version_info(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.client.call("get_version_info"))

    @_record_call
    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool] = None) -> Program:
        self.connect()
        request = NativeQuilRequest(quil=program.out(), target_device=self.target_device)
        response = self.client.call("quil_to_native_quil", request, protoquil=protoquil).asdict()
        nq_program = parse_program(response["quil"])
        nq_program.native_quil_metadata = response["metadata"]
        nq_program.num_shots = program.num_shots
        return nq_program

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> PyQuilExecutableResponse:
        return PyQuilExecutableResponse(
            program=nq_program.out(),
            attributes=_extract_attribute_dictionary_from_program(nq_program),
        )

    @_record_call
    def reset(self) -> None:
        """
        Reset the state of the QVMCompiler quilc connection.
        """
        timeout = self.client.timeout
        self.client.close()  # type: ignore
        self.client = rpcq.Client(self.endpoint, timeout=timeout)

    @property
    def quilc_client(self) -> rpcq.Client:
        """Return the `Client` for the compiler (i.e. quilc, not translation service)."""
        return self.client

    def set_timeout(self, timeout: float) -> None:
        """
        Set timeout for each individual stage of compilation.

        :param timeout: Timeout value for each compilation stage, in seconds. If the stage does not
            complete within this threshold, an exception is raised.
        """
        if timeout < 0:
            raise ValueError(f"Cannot set timeout to negative value {timeout}")

        self.timeout = timeout
        self.quilc_client.rpc_timeout = timeout

    @_record_call
    def get_calibration_program(self) -> Program:
        """
        See ``QPUCompiler.get_calibration_program()``.

        Note: this currently provides an empty Program because the QVM does not support Quil-T.
        """
        return Program()

    @_record_call
    def refresh_calibration_program(self) -> None:
        pass

    @property
    def calibration_program(self) -> Program:
        return Program()


@dataclass
class HTTPCompilerClient:
    """
    A class which partially implements the interface of rpcq.Client, to allow the QPUCompiler to
    send compilation requests over HTTP(S) rather than ZeroMQ.

    :param endpoint: The base url to which rpcq methods will be appended.
    :param session: The ForestSession object which manages headers and authentication.
    """

    endpoint: str
    # session: ForestSession  TODO(andrew): use client here?

    def call(
        self, method: str, payload: Optional[Message] = None, *, rpc_timeout: float = 30
    ) -> Message:
        """
        A partially-compatible implementation of rpcq.Client#call, which allows calling rpcq
        methods over HTTP following the scheme:

            POST <endpoint>/<method>
            body: json-serialized <payload>

        This implementation is meant for use only with the QPUCompiler and is not intended to be
        a fully-compatible port of rpcq from ZeroMQ to HTTP.

        If the request succeeds (per HTTP response code), the body of the response is parsed into
        an RPCQ Message.

        If the request fails, the response body should be a JSON object with a ``message`` field
        indicating the cause of the failure. If present, that message is delivered to the user.

        :param payload: The rpcq message body.
        :param rpc_timeout: The number of seconds to wait for each of 'connection' and 'response'.
            @see https://requests.readthedocs.io/en/master/user/advanced/#timeouts
        """
        url = urljoin(self.endpoint, method)

        if payload:
            body = to_json(payload)  # type: ignore
        else:
            body = None

        response = self.session.post(url, json=body, timeout=rpc_timeout)

        try:
            response.raise_for_status()
        except RequestException as e:
            message = f"QPU Compiler {method} failed: "

            try:
                contents = response.json()
            except Exception:
                contents = None

            if contents and contents.get("message"):
                message += contents.get("message")
            else:
                message += "please try again or contact support."

            raise UserMessageError(message) from e

        return cast(Message, from_json(response.text))  # type: ignore
