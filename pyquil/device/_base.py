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
from typing import List
import numpy as np
from pyquil.contrib.rpcq import CompilerISA, Supported1QGate, Supported2QGate
from pyquil.quilbase import Gate
from pyquil.quilatom import Parameter, unpack_qubit

import networkx as nx


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
    for qubit_id, q in isa.qubits.items():
        if q.dead:
            continue
        for gate in q.gates:
            if gate.operator == Supported1QGate.I:
                gates.append(Gate("I", [], [unpack_qubit(q.id)]))
            elif gate.operator == Supported1QGate.RX:
                gates.extend(
                    [
                        Gate("RX", [np.pi / 2], [unpack_qubit(q.id)]),
                        Gate("RX", [-np.pi / 2], [unpack_qubit(q.id)]),
                        Gate("RX", [np.pi], [unpack_qubit(q.id)]),
                        Gate("RX", [-np.pi], [unpack_qubit(q.id)]),
                    ]
                )
            elif gate.operator == Supported1QGate.RZ:
                gates.append(Gate("RZ", [THETA], [unpack_qubit(q.id)]))
            elif gate.operator == Supported1QGate.WILDCARD:
                gates.extend([Gate("_", "_", [unpack_qubit(q.id)])])
            elif gate.operator == Supported1QGate.MEASURE:
                continue
            else:  # pragma no coverage
                raise ValueError("Unknown qubit gate operator: {}".format(gate.operator))

    for edge_id, e in isa.edges.items():
        if e.dead:
            continue
        targets = [unpack_qubit(t) for t in e.ids]
        for gate in e.gates:
            if gate.operator == Supported2QGate.CZ:
                gates.append(Gate("CZ", [], targets))
                gates.append(Gate("CZ", [], targets[::-1]))
                continue
            if gate.operator == Supported2QGate.ISWAP:
                gates.append(Gate("ISWAP", [], targets))
                gates.append(Gate("ISWAP", [], targets[::-1]))
                continue
            if gate.operator == Supported2QGate.CPHASE:
                gates.append(Gate("CPHASE", [THETA], targets))
                gates.append(Gate("CPHASE", [THETA], targets[::-1]))
                continue
            if gate.operator == Supported2QGate.XY:
                gates.append(Gate("XY", [THETA], targets))
                gates.append(Gate("XY", [THETA], targets[::-1]))
                continue
            if gate.operator == Supported2QGate.WILDCARD:
                gates.append(Gate("_", "_", targets))
                gates.append(Gate("_", "_", targets[::-1]))
                continue

            raise ValueError("Unknown edge type: {}".format(gate.operator))
    return gates