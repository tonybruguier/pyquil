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
from typing import Dict, Any, Optional, List, Tuple

from qcs_api_client.models.get_quilt_calibrations_response import GetQuiltCalibrationsResponse
from qcs_api_client.models.translate_native_quil_to_encrypted_binary_request import (
    TranslateNativeQuilToEncryptedBinaryRequest,
)
from qcs_api_client.operations.sync import (
    translate_native_quil_to_encrypted_binary,
    get_quilt_calibrations,
)
from rpcq.messages import (
    PyQuilExecutableResponse,
    ParameterSpec,
)

from pyquil.api import Client
from pyquil.api._error_reporting import _record_call
from pyquil.api._qac import AbstractCompiler, QuantumExecutable
from pyquil.api._rewrite_arithmetic import rewrite_arithmetic
from pyquil.device._main import AbstractDevice
from pyquil.parser import parse_program
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Measurement, Declare

_log = logging.getLogger(__name__)

PYQUIL_PROGRAM_PROPERTIES = ["native_quil_metadata", "num_shots"]


class QPUCompilerNotRunning(Exception):
    pass


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
    """
    Client to communicate with the compiler and translation service.
    """

    processor_id: str
    _calibration_program: Optional[Program] = None

    @_record_call
    def __init__(
        self,
        *,
        processor_id: str,
        device: AbstractDevice,
        client: Optional[Client] = None,
        timeout: float = 10,
    ) -> None:
        """
        Instantiate a new QPU compiler client.

        :param processor_id: Processor to target.
        :param device: PyQuil Device object to use as compilation target.
        :param client: Optional QCS client. If none is provided, a default client will be created.
        :param timeout: Number of seconds to wait for a response from the client.
        """
        super().__init__(device=device, client=client, timeout=timeout)
        self.processor_id = processor_id

    @_record_call
    def native_quil_to_executable(
        self, nq_program: Program
    ) -> QuantumExecutable:
        arithmetic_response = rewrite_arithmetic(nq_program)  # TODO(andrew): is this still needed?
        request = TranslateNativeQuilToEncryptedBinaryRequest(
            quil=arithmetic_response.quil, num_shots=nq_program.num_shots
        )

        # TODO(andrew): timeout?
        response = self._client.qcs_request(
            translate_native_quil_to_encrypted_binary,
            quantum_processor_id=self.processor_id,
            json_body=request,
        ).parsed

        response.recalculation_table = arithmetic_response.recalculation_table  # type: ignore
        response.memory_descriptors = _collect_memory_descriptors(nq_program)

        # Convert strings to MemoryReference for downstream processing.
        response.ro_sources = [(parse_mref(mref), source) for mref, source in response.ro_sources]

        # TODO (kalzoo): this is a temporary workaround to migrate memory location parsing from
        # the client side (where it was pre-quilt) to the service side. In some cases, the service
        # won't return ro_sources, and so we can fall back to parsing the change on the client side.
        if response.ro_sources == []:
            response.ro_sources = _collect_classical_memory_write_locations(nq_program)

        return response

    def _get_calibration_program(self) -> Program:
        response: GetQuiltCalibrationsResponse = self._client.qcs_request(
            get_quilt_calibrations, quantum_processor_id=self.processor_id
        ).parsed
        return parse_program(response.quilt)

    @_record_call
    @property
    def calibration_program(self) -> Program:
        """
        Get the Quil-T calibration program associated with the underlying QPU.

        A calibration program contains a number of DEFCAL, DEFWAVEFORM, and
        DEFFRAME instructions. In sum, those instructions describe how a Quil-T
        program should be translated into analog instructions for execution on
        hardware.

        :returns: A Program object containing the calibration definitions."""
        if self._calibration_program is None:
            try:
                self._calibration_program = self._get_calibration_program()
            except Exception as ex:
                raise RuntimeError("Could not refresh calibrations") from ex

        return self._calibration_program

    @_record_call
    def reset(self) -> None:
        """
        Reset the state of the QPUCompiler.
        """
        super().reset()
        self._calibration_program = None


class QVMCompiler(AbstractCompiler):
    """
    Client to communicate with the compiler.
    """

    @_record_call
    def __init__(
        self, *, device: AbstractDevice, client: Optional[Client] = None, timeout: float = 10
    ) -> None:
        """
        Client to communicate with compiler.

        :param device: PyQuil Device object to use as compilation target.
        :param client: Optional QCS client. If none is provided, a default client will be created.
        :param timeout: Number of seconds to wait for a response from the client.
        """
        super().__init__(device=device, client=client, timeout=timeout)

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        return PyQuilExecutableResponse(
            program=nq_program.out(),
            attributes=_extract_attribute_dictionary_from_program(nq_program),
        )
