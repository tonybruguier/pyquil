import numpy as np

from pyquil.device import gates_in_isa, THETA
from pyquil.gates import RZ, RX, I, CZ, ISWAP, CPHASE


def test_gates_in_isa(compiler_isa):
    gates = gates_in_isa(compiler_isa)
    for q in [0, 1, 2]:
        for g in [
            I(q),
            RX(np.pi / 2, q),
            RX(-np.pi / 2, q),
            RX(np.pi, q),
            RX(-np.pi, q),
            RZ(THETA, q),
        ]:
            assert g in gates

    assert CZ(0, 1) in gates
    assert CZ(1, 0) in gates
    assert ISWAP(1, 2) in gates
    assert ISWAP(2, 1) in gates
    assert CPHASE(THETA, 2, 0) in gates
    assert CPHASE(THETA, 0, 2) in gates
