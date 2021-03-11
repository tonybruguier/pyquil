from pyquil.external.rpcq import (
    make_edge_id,
)
from pyquil.device import QCSDevice
from pyquil.device.transformers import qcs_isa_to_compiler_isa
from pyquil.noise import NoiseModel


def test_qcs_isa_to_compiler_isa(qcs_aspen8_isa, aspen8_compiler_isa):
    compiler_isa = qcs_isa_to_compiler_isa(qcs_aspen8_isa)

    for node in qcs_aspen8_isa.architecture.nodes:
        assert str(node.node_id) in compiler_isa.qubits

    for edge in qcs_aspen8_isa.architecture.edges:
        assert make_edge_id(edge.node_ids[0], edge.node_ids[1]) in compiler_isa.edges

    assert compiler_isa == aspen8_compiler_isa


def test_qcs_noise_model(qcs_aspen8_isa, noise_model_dict):
    noise_model = NoiseModel.from_dict(noise_model_dict)
    device = QCSDevice("Aspen-8", qcs_aspen8_isa, noise_model=noise_model)
    assert device.quantum_processor_id == "Aspen-8"

    assert isinstance(device.noise_model, NoiseModel)
    assert device.noise_model == noise_model
