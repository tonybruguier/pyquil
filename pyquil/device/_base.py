##############################################################################
# Copyright 2016-2019 Rigetti Computing
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
from typing import List, Iterable
from pyquil.external.rpcq import CompilerISA, Supported1QGate, Supported2QGate, GateInfo
from pyquil.quilbase import Gate, ParameterDesignator
from pyquil.quilatom import Parameter, unpack_qubit
import logging

import networkx as nx

_log = logging.getLogger(__name__)
THETA = Parameter("theta")


class AbstractDevice(ABC):
    @abstractmethod
    def qubits(self) -> List[int]:
        """
        A sorted list of qubits in the device topology.
        """

    @abstractmethod
    def qubit_topology(self) -> nx.Graph:
        """
        The connectivity of qubits in this device given as a NetworkX graph.
        """

    @abstractmethod
    def to_compiler_isa(self) -> CompilerISA:
        """
        Construct an ISA suitable for targeting by compilation.
        This will raise an exception if the requested ISA is not supported by the device.
        """


def gates_in_isa(isa: CompilerISA) -> List[Gate]:
    """
    Generate the full gateset associated with an ISA.
    :param isa: The instruction set architecture for a QPU.
    :return: A sequence of Gate objects encapsulating all gates compatible with the ISA.
    """
    gates = []
    for _qubit_id, q in isa.qubits.items():
        if q.dead:
            continue
        for gate in q.gates:
            if gate.operator in {Supported1QGate.I, Supported1QGate.RX, Supported1QGate.RZ}:
                # FIXME  this is ugly
                assert isinstance(gate, GateInfo)
                if len(gate.parameters) > 0 and gate.parameters[0] == 0.0:
                    continue
                parameters: Iterable[ParameterDesignator] = []
                if (
                    gate.operator == Supported1QGate.RZ
                    and len(gate.parameters) == 1
                    and gate.parameters[0] == "_"
                ):
                    parameters = [Parameter("theta")]
                else:
                    parameters = [
                        Parameter(param) if isinstance(param, str) else param
                        for param in gate.parameters
                    ]
                gates.append(Gate(gate.operator, parameters, [unpack_qubit(q.id)]))
            elif gate.operator == Supported1QGate.MEASURE:
                continue
            else:  # pragma no coverage
                raise ValueError("Unknown qubit gate operator: {}".format(gate.operator))

    for edge_id, e in isa.edges.items():
        if e.dead:
            continue

        # QUESTION: This was previously non-comprehensive, is that still the approach we
        # want to take? eg gates of [XY, CZ] only yields gates of [CZ]
        operators = [gate.operator for gate in e.gates]
        targets = [unpack_qubit(t) for t in e.ids]
        if Supported2QGate.CZ in operators:
            gates.append(Gate("CZ", [], targets))
            gates.append(Gate("CZ", [], targets[::-1]))
            continue
        if Supported2QGate.ISWAP in operators:
            gates.append(Gate("ISWAP", [], targets))
            gates.append(Gate("ISWAP", [], targets[::-1]))
            continue
        if Supported2QGate.CPHASE in operators:
            gates.append(Gate("CPHASE", [THETA], targets))
            gates.append(Gate("CPHASE", [THETA], targets[::-1]))
            continue
        if Supported2QGate.XY in operators:
            gates.append(Gate("XY", [THETA], targets))
            gates.append(Gate("XY", [THETA], targets[::-1]))
            continue
        if Supported2QGate.WILDCARD in operators:
            gates.append(Gate("_", "_", targets))
            gates.append(Gate("_", "_", targets[::-1]))
            continue

        _log.warning(f"no gate for edge {edge_id}")
    return gates
