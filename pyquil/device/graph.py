from typing import Any, Tuple, Union

import numpy as np
from pyquil.device._base import AbstractDevice
from typing import List, Optional, cast
from pyquil.external.rpcq import (
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
    """
    An AbstractDevice initialized with a user constructed NetworkX graph topology.
    Notably, this class is able to serialize a ``CompilerISA`` based on the
    graph topology and the configured 1Q and 2Q gates.
    """

    def __init__(
        self,
        topology: nx.Graph,
        gates_1q: Optional[List[str]] = None,
        gates_2q: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a new NxDevice.

        :param topology: The graph topology of the device.
        :param gates_1q: A list of 1Q gate names supported by all qubits in the device.
        :param gates_2q: A list of 2Q gate names supported all edges in the device.
        """
        self.topology = topology
        self.gates_1q = gates_1q
        self.gates_2q = gates_2q

    def qubit_topology(self) -> nx.Graph:
        return self.topology

    def to_compiler_isa(self) -> CompilerISA:
        """
        Generate a ``CompilerISA`` object based on a NetworkX graph and the
        ``gates_1q`` and ``gates_2q`` with which the device was initialized.

        May raise ``GraphGateError`` if the specified gates are not supported.
        """
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
    Generate an ``CompilerISA`` object from a NetworkX graph and list of 1Q and 2Q gates.
    May raise ``GraphGateError`` if the specified gates are not supported.

    :param graph: The graph topology of the device.
    :param gates_1q: A list of 1Q gate names to be applied to all qubits in the device.
    Defaults to ``DEFAULT_1Q_GATES``.
    :param gates_2q: A list of 2Q gate names to be applied to all edges in the device.
    Defaults to ``DEFAULT_2Q_GATES``.
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


def compiler_isa_to_graph(compiler_isa: CompilerISA) -> nx.Graph:
    """
    Generate an ``nx.Graph`` based on the qubits and edges of any ``CompilerISA``.
    """
    return nx.from_edgelist([int(i) for i in edge.ids] for edge in compiler_isa.edges.values())


class GraphGateError(ValueError):
    """
    Signals an error when creating a ``CompilerISA`` from an ``nx.Graph``.
    This may raise as a consequence of unsupported gates.
    """

    pass


def _make_i_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported1QGate.I, parameters=[], arguments=["_"])]


def _make_measure_gates() -> List[MeasureInfo]:
    return [
        MeasureInfo(operator=Supported1QGate.MEASURE, qubit="_", target="_"),
        MeasureInfo(operator=Supported1QGate.MEASURE, qubit="_", target=None),
    ]


def _make_rx_gates() -> List[GateInfo]:
    gates = []
    for param in [0.0, np.pi, -np.pi, np.pi / 2, -np.pi / 2]:
        gates.append(GateInfo(operator=Supported1QGate.RX, parameters=[param], arguments=["_"]))
    return gates


def _make_rz_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported1QGate.RZ, parameters=["theta"], arguments=["_"])]


def _make_wildcard_1q_gates() -> List[GateInfo]:
    return [GateInfo(operator="_", parameters=["_"], arguments=["_"])]


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
        raise GraphGateError("Unsupported graph qubit operation: {}".format(operation_name))


def _make_cz_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.CZ, parameters=[], arguments=["_", "_"])]


def _make_iswap_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.ISWAP, parameters=[], arguments=["_", "_"])]


def _make_cphase_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.CPHASE, parameters=["theta"], arguments=["_", "_"])]


def _make_xy_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported2QGate.XY, parameters=["theta"], arguments=["_", "_"])]


def _make_wildcard_2q_gates() -> List[GateInfo]:
    return [GateInfo(operator="_", parameters=["_"], arguments=["_", "_"])]


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
        raise GraphGateError("Unsupported graph edge operation: {}".format(operation_name))
