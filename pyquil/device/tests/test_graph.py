import networkx as nx
from pyquil.device.transformers.graph_to_compiler_isa import (
    compiler_isa_to_graph,
    DEFAULT_2Q_GATES,
    DEFAULT_1Q_GATES,
)
from pyquil.device.transformers import graph_to_compiler_isa
from pyquil.device.graph import NxDevice


def test_isa_from_graph_order():
    # since node 16 appears first, even though we ask for the edge (15,16) the networkx internal
    # representation will have it as (16,15)
    fc = nx.from_edgelist([(16, 17), (15, 16)])
    isa = graph_to_compiler_isa(fc)
    isad = isa.dict(by_alias=True)
    for k in isad["2Q"]:
        q1, q2 = k.split("-")
        assert q1 < q2


def test_compiler_isa_to_graph(compiler_isa):
    graph = compiler_isa_to_graph(compiler_isa)
    should_be = nx.from_edgelist([(0, 1), (1, 2), (0, 2), (0, 3)])
    assert nx.is_isomorphic(graph, should_be)

    nx_device = NxDevice(graph)
    assert nx.is_isomorphic(graph, nx_device.qubit_topology())


def test_graph_to_compiler_isa(compiler_isa, noise_model_dict):
    graph = compiler_isa_to_graph(compiler_isa)
    isa = graph_to_compiler_isa(graph)

    for _, qubit in isa.qubits.items():
        for gate in DEFAULT_1Q_GATES:
            assert any([gate == qubit_gate.operator for qubit_gate in qubit.gates])

    for _, edge in isa.edges.items():
        for gate in DEFAULT_2Q_GATES:
            assert any([gate == edge_gate.operator for edge_gate in edge.gates])
