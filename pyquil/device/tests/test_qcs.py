import numpy as np

from pyquil.device import QCSDevice
from pyquil.noise import NoiseModel
from pyquil.gates import RZ, RX, CZ
from pyquil.noise_gates import _get_qvm_noise_supported_gates, THETA

DEVICE_FIXTURE_NAME = "mixed_architecture_chip"


ASPEN_8_QUBITS_NO_RX = {8, 9, 10, 18, 19, 28, 29, 31}
ASPEN_8_QUBITS_NO_RZ = {8, 9, 10, 18, 19, 28, 29, 31}
ASPEN_8_EDGES_NO_CZ = {(0, 1), (10, 11), (1, 2), (21, 22), (17, 10), (12, 25)}


def test_qcs_device(qcs_aspen8_isa, noise_model_dict):
    noise_model = NoiseModel.from_dict(noise_model_dict)
    device = QCSDevice(DEVICE_FIXTURE_NAME, qcs_aspen8_isa, noise_model=noise_model)
    assert device.quantum_processor_id == DEVICE_FIXTURE_NAME

    assert isinstance(device.noise_model, NoiseModel)
    assert device.noise_model == noise_model

    compiler_isa = device.to_compiler_isa()
    gates = _get_qvm_noise_supported_gates(compiler_isa)

    for q in range(len(device._isa.architecture.nodes)):
        if q not in ASPEN_8_QUBITS_NO_RX:
            for g in [
                RX(np.pi / 2, q),
                RX(-np.pi / 2, q),
                RX(np.pi, q),
                RX(-np.pi, q),
            ]:
                assert g in gates
        if q not in ASPEN_8_QUBITS_NO_RZ:
            assert RZ(THETA, q) in gates

    for edge in device._isa.architecture.edges:
        if (edge.node_ids[0], edge.node_ids[1],) in ASPEN_8_EDGES_NO_CZ:
            continue
        assert CZ(edge.node_ids[0], edge.node_ids[1]) in gates
        assert CZ(edge.node_ids[1], edge.node_ids[0]) in gates
