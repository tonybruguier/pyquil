from typing import Any, Tuple, Union

import numpy as np
from pyquil.device._base import AbstractDevice
from typing import List, Optional, cast
from pyquil.contrib.rpcq import (
    GateInfo,
    MeasureInfo,
    Supported1QGate,
    Supported2QGate,
    CompilerISA,
    add_qubit,
    add_edge,
)

import networkx as nx


class NxDevice(AbstractDevice):
    """A shim over the AbstractDevice API backed by a NetworkX graph.

    A ``Device`` holds information about the physical device.
    Specifically, you might want to know about connectivity, available gates, performance specs,
    and more. This class implements the AbstractDevice API for devices not available via
    ``get_devices()``. Instead, the user is responsible for constructing a NetworkX
    graph which represents a chip topology.
    """

    def __init__(
        self,
        topology: nx.Graph,
        gates_1q: Optional[List[str]] = None,
        gates_2q: Optional[List[str]] = None,
    ) -> None:
        self.topology = topology
        self.gates_1q = gates_1q
        self.gates_2q = gates_2q

    def qubit_topology(self) -> nx.Graph:
        return self.topology

    def to_compiler_isa(self) -> CompilerISA:
        return compiler_isa_from_graph(
            self.topology, gates_1q=self.gates_1q, gates_2q=self.gates_2q
        )

    def qubits(self) -> List[int]:
        return sorted(self.topology.nodes)

    def edges(self) -> List[Tuple[Any, ...]]:
        return sorted(tuple(sorted(pair)) for pair in self.topology.edges)


DEFAULT_1Q_GATES = [
    Supported1QGate.I,
    Supported1QGate.RX,
    Supported1QGate.RZ,
    Supported1QGate.MEASURE,
]
DEFAULT_2Q_GATES = [
    Supported2QGate.CZ,
    Supported2QGate.XY,
]


def compiler_isa_from_graph(
    graph: nx.Graph, gates_1q: Optional[List[str]] = None, gates_2q: Optional[List[str]] = None
) -> CompilerISA:
    """
    Generate an ISA object from a NetworkX graph.
    """
    gates_1q = gates_1q or DEFAULT_1Q_GATES.copy()
    gates_2q = gates_2q or DEFAULT_2Q_GATES.copy()

    device = CompilerISA()

    qubit_gates = []
    for gate in gates_1q:
        qubit_gates.extend(_transform_qubit_operation_to_gates(gate))

    all_qubits = list(range(max(graph.nodes) + 1))
    for i in all_qubits:
        qubit = add_qubit(device, i)
        qubit.gates = qubit_gates
        qubit.dead = i not in graph.nodes

    edge_gates = []
    for gate in gates_2q:
        edge_gates.extend(_transform_edge_operation_to_gates(gate))

    for a, b in graph.edges:
        edge = add_edge(device, a, b)
        edge.gates = edge_gates

    return device


def compiler_isa_to_graph(device: CompilerISA) -> nx.Graph:
    return nx.from_edgelist([int(i) for i in edge.ids] for edge in device.edges.values())


def _make_i_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported1QGate.I, parameters=[], arguments=["_"])]


def _make_measure_gates() -> List[MeasureInfo]:
    return [
        MeasureInfo(operator=Supported1QGate.MEASURE, qubit="_", target="_"),
        MeasureInfo(operator=Supported1QGate.MEASURE, qubit="_", target=None),
    ]


def _make_rx_gates() -> List[GateInfo]:
    gates = [GateInfo(operator=Supported1QGate.RX, parameters=[0.0], arguments=["_"])]
    for param in [np.pi, -np.pi, np.pi / 2, -np.pi / 2]:
        gates.append(GateInfo(operator=Supported1QGate.RX, parameters=[param], arguments=["_"]))
    return gates


def _make_rz_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported1QGate.RZ, parameters=["theta"], arguments=["_"])]


def _make_wildcard_1q_gates() -> List[GateInfo]:
    return [GateInfo(operator="_", parameters="_", arguments=["_"])]


def _transform_qubit_operation_to_gates(operation_name: str,) -> List[Union[GateInfo, MeasureInfo]]:
    if operation_name == Supported1QGate.I:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_i_gates())
    elif operation_name == Supported1QGate.RX:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_rx_gates())
    elif operation_name == Supported1QGate.RZ:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_rz_gates())
    elif operation_name == Supported1QGate.MEASURE:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_measure_gates())
    elif operation_name == Supported1QGate.WILDCARD:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_wildcard_1q_gates())
    else:
        # QUESTION: Log error here? Include parameter for hard or soft failure?
        raise ValueError("Unknown qubit operation: {}".format(operation_name))


def _make_cz_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.CZ, parameters=[], arguments=["_", "_"])]


def _make_iswap_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.ISWAP, parameters=[], arguments=["_", "_"])]


def _make_cphase_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.CPHASE, parameters=["theta"], arguments=["_", "_"])]


def _make_xy_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.XY, parameters=["theta"], arguments=["_", "_"])]


def _make_wildcard_2q_gates() -> List[GateInfo]:
    return [GateInfo(operator="_", parameters="_", arguments=["_", "_"])]


def _transform_edge_operation_to_gates(operation_name: str) -> List[GateInfo]:
    if operation_name == Supported2QGate.CZ:
        return _make_cz_gates()
    elif operation_name == Supported2QGate.ISWAP:
        return _make_iswap_gates()
    elif operation_name == Supported2QGate.CPHASE:
        return _make_cphase_gates()
    elif operation_name == Supported2QGate.XY:
        return _make_xy_gates()
    elif operation_name == Supported2QGate.WILDCARD:
        return _make_wildcard_2q_gates()
    else:
        raise ValueError("Unknown edge operation: {}".format(operation_name))
