from qcs_api_client.models import InstructionSetArchitecture, Characteristic
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
from pyquil.device.graph import AbstractDevice
from pyquil.noise import NoiseModel
import networkx as nx
import numpy as np
from pyquil.external.rpcq import (
    GateInfo,
    MeasureInfo,
    Supported1QGate,
    Supported2QGate,
    make_edge_id,
)
from typing import List, Union, Optional, cast, DefaultDict, Set
from collections import defaultdict


class QCSDevice(AbstractDevice):
    """
    An AbstractDevice initialized with an ``InstructionSetArchitecture`` returned
    from the QCS API. Notably, this class is able to serialize a ``CompilerISA`` based
    on the architecture instructions.
    """

    quantum_processor_id: str
    _isa: InstructionSetArchitecture
    noise_model: Optional[NoiseModel]

    def __init__(
        self,
        quantum_processor_id: str,
        isa: InstructionSetArchitecture,
        noise_model: Optional[NoiseModel] = None,
    ):
        """
        Initialize a new QCSDevice.

        :param quantum_processor_id: The id of the quantum processor.
        :param isa: The QCS API ``InstructionSetArchitecture``.
        :param noise_model: An optional ``NoiseModel`` for configuring a noisy device
        on the ``QVM``.
        """

        self.quantum_processor_id = quantum_processor_id
        self._isa = isa
        self.noise_model = noise_model

    def qubits(self) -> List[int]:
        return sorted(node.node_id for node in self._isa.architecture.nodes)

    def qubit_topology(self) -> nx.Graph:
        return nx.from_edgelist(edge.node_ids for edge in self._isa.architecture.edges)

    def to_compiler_isa(self) -> CompilerISA:
        return _transform_qcs_isa_to_compiler_isa(self._isa)

    def __str__(self) -> str:
        return "<QCSDevice {}>".format(self.quantum_processor_id)

    def __repr__(self) -> str:
        return str(self)


class QCSISAParseError(ValueError):
    """
    Signals an error when creating a ``CompilerISA`` due to the operators
    in the QCS ``InstructionSetArchitecture``. This may raise as a consequence
    of unsupported gates as well as missing nodes or edges.
    """

    pass


def _transform_qcs_isa_to_compiler_isa(isa: InstructionSetArchitecture) -> CompilerISA:
    device = CompilerISA()
    for node in isa.architecture.nodes:
        add_qubit(device, node.node_id)

    for edge in isa.architecture.edges:
        add_edge(device, edge.node_ids[0], edge.node_ids[1])

    qubit_operations_seen: DefaultDict[int, Set[str]] = defaultdict(set)
    edge_operations_seen: DefaultDict[str, Set[str]] = defaultdict(set)
    for operation in isa.instructions:
        for site in operation.sites:
            if operation.node_count == 1:
                if len(site.node_ids) != 1:
                    raise QCSISAParseError(
                        f"operation {operation.name} has node count 1, but "
                        f"site has {len(site.node_ids)} node_ids"
                    )
                operation_qubit = get_qubit(device, site.node_ids[0])
                if operation_qubit is None:
                    raise QCSISAParseError(
                        f"operation {operation.name} has node {site.node_ids[0]} "
                        f"but node not declared in architecture"
                    )

                if operation.name in qubit_operations_seen[operation_qubit.id]:
                    continue
                qubit_operations_seen[operation_qubit.id].add(operation.name)

                operation_qubit.gates.extend(
                    _transform_qubit_operation_to_gates(
                        operation.name, operation_qubit.id, site.characteristics
                    )
                )

            elif operation.node_count == 2:
                if len(site.node_ids) != 2:
                    QCSISAParseError(
                        f"operation {operation.name} has node count 2, but site "
                        f"has {len(site.node_ids)} node_ids"
                    )

                operation_edge = get_edge(device, site.node_ids[0], site.node_ids[1])
                edge_id = make_edge_id(site.node_ids[0], site.node_ids[1])
                if operation_edge is None:
                    raise QCSISAParseError(
                        f"operation {operation.name} has site {site.node_ids}, but edge {edge_id} "
                        f"not declared in architecture"
                    )

                if operation.name in edge_operations_seen[edge_id]:
                    continue
                edge_operations_seen[edge_id].add(operation.name)

                operation_edge.gates.extend(
                    _transform_edge_operation_to_gates(operation.name, site.characteristics)
                )

            else:
                raise QCSISAParseError(
                    "unexpected operation node count: {}".format(operation.node_count)
                )

    return device


PERFECT_FIDELITY = 1e0
PERFECT_DURATION = 1 / 100

_operation_names_to_compiler_fidelity_default = {
    Supported2QGate.CZ: 0.89,
    Supported2QGate.ISWAP: 0.90,
    Supported2QGate.CPHASE: 0.85,
    Supported2QGate.XY: 0.86,
    Supported1QGate.RX: 0.95,
    Supported1QGate.MEASURE: 0.90,
}

_operation_names_to_compiler_duration_default = {
    Supported2QGate.CZ: 200,
    Supported2QGate.ISWAP: 200,
    Supported2QGate.CPHASE: 200,
    Supported2QGate.XY: 200,
    Supported1QGate.RX: 50,
    Supported1QGate.MEASURE: 2000,
}


def _make_measure_gates(node_id: int, characteristics: List[Characteristic]) -> List[MeasureInfo]:
    duration = _operation_names_to_compiler_duration_default[Supported1QGate.MEASURE]
    fidelity = _operation_names_to_compiler_fidelity_default[Supported1QGate.MEASURE]
    for characteristic in characteristics:
        if characteristic.name == "fRO":
            fidelity = characteristic.value
            break

    return [
        MeasureInfo(
            operator=Supported1QGate.MEASURE,
            qubit=str(node_id),
            target="_",
            fidelity=fidelity,
            duration=duration,
        ),
        MeasureInfo(
            operator=Supported1QGate.MEASURE,
            qubit=str(node_id),
            target=None,
            fidelity=fidelity,
            duration=duration,
        ),
    ]


def _make_rx_gates(node_id: int, characteristics: List[Characteristic]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported1QGate.RX]

    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported1QGate.RX]
    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "f1QRB":
            fidelity = characteristic.value
            break

    gates = [
        GateInfo(
            operator=Supported1QGate.RX,
            parameters=[0.0],
            arguments=[node_id],
            fidelity=PERFECT_FIDELITY,
            duration=default_duration,
        )
    ]
    for param in [np.pi, -np.pi, np.pi / 2, -np.pi / 2]:
        gates.append(
            GateInfo(
                operator=Supported1QGate.RX,
                parameters=[param],
                arguments=[node_id],
                fidelity=fidelity,
                duration=default_duration,
            )
        )
    return gates


def _make_rz_gates(node_id: int) -> List[GateInfo]:
    return [
        GateInfo(
            operator=Supported1QGate.RZ,
            parameters=["_"],
            arguments=[node_id],
            fidelity=PERFECT_FIDELITY,
            duration=PERFECT_DURATION,
        )
    ]


def _make_wildcard_1q_gates(node_id: int) -> List[GateInfo]:
    return [
        GateInfo(
            operator="_",
            parameters=["_"],
            arguments=[node_id],
            fidelity=PERFECT_FIDELITY,
            duration=PERFECT_DURATION,
        )
    ]


def _transform_qubit_operation_to_gates(
    operation_name: str, node_id: int, characteristics: List[Characteristic],
) -> List[Union[GateInfo, MeasureInfo]]:
    if operation_name == Supported1QGate.RX:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_rx_gates(node_id, characteristics))
    elif operation_name == Supported1QGate.RZ:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_rz_gates(node_id))
    elif operation_name == Supported1QGate.MEASURE:
        return cast(
            List[Union[GateInfo, MeasureInfo]], _make_measure_gates(node_id, characteristics)
        )
    elif operation_name == Supported1QGate.WILDCARD:
        return cast(List[Union[GateInfo, MeasureInfo]], _make_wildcard_1q_gates(node_id))
    elif operation_name in {"I", "RESET"}:
        # QUESTION: Doesn't seem to be existing support for reset operation
        return []
    else:
        raise QCSISAParseError("Unsupported qubit operation: {}".format(operation_name))


def _make_cz_gates(characteristics: List[Characteristic]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.CZ]

    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.CZ]
    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fCZ":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.CZ,
            parameters=[],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_iswap_gates(characteristics: List[Characteristic]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.ISWAP]

    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.ISWAP]
    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fISWAP":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.ISWAP,
            parameters=[],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_cphase_gates(characteristics: List[Characteristic]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.CPHASE]

    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.CPHASE]
    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fCPHASE":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.ISWAP,
            parameters=["theta"],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_xy_gates(characteristics: List[Characteristic]) -> List[GateInfo]:
    default_duration = _operation_names_to_compiler_duration_default[Supported2QGate.XY]

    default_fidelity = _operation_names_to_compiler_fidelity_default[Supported2QGate.XY]
    fidelity = default_fidelity
    for characteristic in characteristics:
        if characteristic.name == "fXY":
            fidelity = characteristic.value
            break

    return [
        GateInfo(
            operator=Supported2QGate.XY,
            parameters=["theta"],
            arguments=["_", "_"],
            fidelity=fidelity,
            duration=default_duration,
        )
    ]


def _make_wildcard_2q_gates() -> List[GateInfo]:
    return [
        GateInfo(
            operator="_",
            parameters=["_"],
            arguments=["_", "_"],
            fidelity=PERFECT_FIDELITY,
            duration=PERFECT_DURATION,
        )
    ]


def _transform_edge_operation_to_gates(
    operation_name: str, characteristics: List[Characteristic],
) -> List[GateInfo]:
    if operation_name == Supported2QGate.CZ:
        return _make_cz_gates(characteristics)
    elif operation_name == Supported2QGate.ISWAP:
        return _make_iswap_gates(characteristics)
    elif operation_name == Supported2QGate.CPHASE:
        return _make_cphase_gates(characteristics)
    elif operation_name == Supported2QGate.XY:
        return _make_xy_gates(characteristics)
    elif operation_name == Supported2QGate.WILDCARD:
        return _make_wildcard_2q_gates()
    else:
        raise QCSISAParseError("Unsupported edge operation: {}".format(operation_name))
