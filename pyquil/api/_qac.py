##############################################################################
# Copyright 2018 Rigetti Computing
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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

from rpcq.messages import (
    NativeQuilRequest,
    TargetDevice, QuiltBinaryExecutableResponse, PyQuilExecutableResponse,
)

from pyquil.api import Client
from pyquil.api._error_reporting import _record_call
from pyquil.device import AbstractDevice
from pyquil.parser import parse_program
from pyquil.paulis import PauliTerm
from pyquil.quil import Program
from pyquil.quilbase import Gate
from pyquil.version import __version__


class QuilcVersionMismatch(Exception):
    pass


class QuilcNotRunning(Exception):
    pass


QuantumExecutable = Union[QuiltBinaryExecutableResponse, PyQuilExecutableResponse]


class AbstractCompiler(ABC):
    """The abstract interface for a compiler."""

    _target_device: TargetDevice
    _client: Optional[Client]
    _timeout: float

    def __init__(
        self, *, device: AbstractDevice, client: Optional[Client], timeout: float
    ) -> None:
        self._target_device = TargetDevice(isa=device.get_isa().to_dict(), specs={})
        self._client = client or Client()

        if not self._client.quilc_url.startswith("tcp://"):
            raise ValueError(
                f"Expected compiler URL '{self._client.quilc_url}' to start with 'tcp://'"
            )
        self.set_timeout(timeout)

    def get_version_info(self) -> Dict[str, Any]:
        """
        Return version information for this compiler and its dependencies.

        :return: Dictionary of version information.
        """
        quilc_version_info = self._client.compiler_rpcq_request(
            "get_version_info",
            timeout=self._timeout,
        )
        return {"quilc": quilc_version_info}

    @_record_call
    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool] = None) -> Program:
        """
        Compile an arbitrary quil program according to the ISA of target_device.

        :param program: Arbitrary quil to compile
        :param protoquil: Whether to restrict to protoquil (``None`` means defer to server)
        :return: Native quil and compiler metadata
        """
        self._connect()
        request = NativeQuilRequest(
            quil=program.out(calibrations=False), target_device=self._target_device
        )
        response = self._client.compiler_rpcq_request(
            "quil_to_native_quil",
            request,
            protoquil=protoquil,
            timeout=self._timeout,
        ).asdict()
        nq_program = parse_program(response["quil"])
        nq_program.native_quil_metadata = response["metadata"]
        nq_program.num_shots = program.num_shots
        nq_program._calibrations = program.calibrations
        return nq_program

    def _connect(self) -> None:
        try:
            quilc_version_dict = self._client.compiler_rpcq_request("get_version_info", timeout=self._timeout)
            _check_quilc_version(quilc_version_dict)
        except TimeoutError:
            raise QuilcNotRunning(
                f"Request to quilc at {self._client.quilc_url} timed out. "
                "This could mean that quilc is not running, is not reachable, or is "
                "responding slowly."
            )

    @abstractmethod
    def native_quil_to_executable(
        self, nq_program: Program
    ) -> QuantumExecutable:
        """
        Compile a native quil program to a binary executable.

        :param nq_program: Native quil to compile
        :return: An (opaque) binary executable
        """

    def set_timeout(self, timeout: float) -> None:
        """
        Set timeout for each individual stage of compilation.

        :param timeout: Timeout value for each compilation stage, in seconds. If the stage does not
            complete within this threshold, an exception is raised.
        """
        if timeout < 0:
            raise ValueError(f"Cannot set timeout to negative value {timeout}")

        self._timeout = timeout

    @_record_call
    def reset(self) -> None:
        """
        Reset the state of the this compiler client.
        """
        pass


def _check_quilc_version(version_dict: Dict[str, str]) -> None:
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


class AbstractBenchmarker(ABC):
    @abstractmethod
    def apply_clifford_to_pauli(self, clifford: Program, pauli_in: PauliTerm) -> PauliTerm:
        r"""
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing PCP^{\dagger}.

        :param clifford: A Program that consists only of Clifford operations.
        :param pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to pauli_in * clifford * pauli_in^{\dagger}
        """

    @abstractmethod
    def generate_rb_sequence(
        self,
        depth: int,
        gateset: Sequence[Gate],
        seed: Optional[int] = None,
        interleaver: Optional[Program] = None,
    ) -> List[Program]:
        """
        Construct a randomized benchmarking experiment on the given qubits, decomposing into
        gateset. If interleaver is not provided, the returned sequence will have the form

            C_1 C_2 ... C_(depth-1) C_inv ,

        where each C is a Clifford element drawn from gateset, C_{< depth} are randomly selected,
        and C_inv is selected so that the entire sequence composes to the identity.  If an
        interleaver G (which must be a Clifford, and which will be decomposed into the native
        gateset) is provided, then the sequence instead takes the form

            C_1 G C_2 G ... C_(depth-1) G C_inv .

        The JSON response is a list of lists of indices, or Nones. In the former case, they are the
        index of the gate in the gateset.

        :param int depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param list gateset: A list of pyquil gates to decompose the Clifford elements into. These
         must generate the clifford group on the qubits of interest. e.g. for one qubit
         [RZ(np.pi/2), RX(np.pi/2)].
        :param seed: A positive integer used to seed the PRNG.
        :param interleaver: A Program object that encodes a Clifford element.
        :return: A list of pyquil programs. Each pyquil program is a circuit that represents an
         element of the Clifford group. When these programs are composed, the resulting Program
         will be the randomized benchmarking experiment of the desired depth. e.g. if the return
         programs are called cliffords then `sum(cliffords, Program())` will give the randomized
         benchmarking experiment, which will compose to the identity program.
        """
