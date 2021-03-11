import numpy as np
import pytest
import json
import os
from pathlib import Path
from requests import RequestException

from qcs_api_client.client._configuration import QCSClientConfiguration
from qcs_api_client.models import InstructionSetArchitecture
from pyquil.api import (
    QVMConnection,
    QVMCompiler,
    Client,
    BenchmarkConnection,
)
from pyquil.device import QCSDevice, CompilerDevice
from pyquil.device.graph import (
    DEFAULT_1Q_GATES, DEFAULT_2Q_GATES, _transform_edge_operation_to_gates, _transform_qubit_operation_to_gates,
)
from pyquil.api._errors import UnknownApiError
from pyquil.api._abstract_compiler import QuilcNotRunning, QuilcVersionMismatch
from pyquil.api._qvm import QVMNotRunning, QVMVersionMismatch
from pyquil.external.rpcq import CompilerISA, GateInfo
from pyquil.gates import I
from pyquil.paulis import sX
from pyquil.quil import Program
from pyquil.tests.utils import DummyCompiler


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def compiler_isa():
    gates_1q = []
    for gate in DEFAULT_1Q_GATES:
        gates_1q.extend(_transform_qubit_operation_to_gates(gate))
    gates_2q = []
    for gate in DEFAULT_2Q_GATES:
        gates_2q.extend(_transform_edge_operation_to_gates(gate))
    isa = CompilerISA.parse_obj({
        "1Q": {
            "0": {"id": 0, "gates": gates_1q},
            "1": {"id": 1, "gates": gates_1q},
            "2": {"id": 2, "gates": gates_1q},
            "3": {"id": 3, "dead": True}
        },
        "2Q": {
            "0-1": {"ids": [0, 1], "gates": gates_2q},
            "1-2": {"ids": [1, 2], "gates": [GateInfo(operator="ISWAP", parameters=[], arguments=["_", "_"])]},
            "0-2": {"ids": [0, 2], "gates": [GateInfo(operator="CPHASE", parameters=["theta"], arguments=["_", "_"])]},
            "0-3": {"ids": [0, 3], "dead": True},
        },
    })
    return isa


@pytest.fixture()
def compiler_device(compiler_isa):
    return CompilerDevice(isa=compiler_isa)


@pytest.fixture
def qcs_aspen8_isa():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f"{dir_path}/pyquil/tests/data/qcs-isa-Aspen-8.json") as f:
        return InstructionSetArchitecture.from_dict(json.load(f))


@pytest.fixture
def specs_dict():
    return {
        "1Q": {
            "0": {
                "f1QRB": 0.99,
                "f1QRB_std_err": 0.01,
                "f1Q_simultaneous_RB": 0.98,
                "f1Q_simultaneous_RB_std_err": 0.02,
                "fRO": 0.93,
                "T1": 20e-6,
                "T2": 15e-6,
            },
            "1": {
                "f1QRB": 0.989,
                "f1QRB_std_err": 0.011,
                "f1Q_simultaneous_RB": 0.979,
                "f1Q_simultaneous_RB_std_err": 0.021,
                "fRO": 0.92,
                "T1": 19e-6,
                "T2": 12e-6,
            },
            "2": {
                "f1QRB": 0.983,
                "f1QRB_std_err": 0.017,
                "f1Q_simultaneous_RB": 0.973,
                "f1Q_simultaneous_RB_std_err": 0.027,
                "fRO": 0.95,
                "T1": 21e-6,
                "T2": 16e-6,
            },
            "3": {
                "f1QRB": 0.988,
                "f1QRB_std_err": 0.012,
                "f1Q_simultaneous_RB": 0.978,
                "f1Q_simultaneous_RB_std_err": 0.022,
                "fRO": 0.94,
                "T1": 18e-6,
                "T2": 11e-6,
            },
        },
        "2Q": {
            "0-1": {"fBellState": 0.90, "fCZ": 0.89, "fCZ_std_err": 0.01, "fCPHASE": 0.88},
            "1-2": {"fBellState": 0.91, "fCZ": 0.90, "fCZ_std_err": 0.12, "fCPHASE": 0.89},
            "0-2": {"fBellState": 0.92, "fCZ": 0.91, "fCZ_std_err": 0.20, "fCPHASE": 0.90},
            "0-3": {"fBellState": 0.89, "fCZ": 0.88, "fCZ_std_err": 0.03, "fCPHASE": 0.87},
        },
    }


@pytest.fixture
def noise_model_dict():
    return {
        "gates": [
            {
                "gate": "I",
                "params": (5.0,),
                "targets": (0, 1),
                "kraus_ops": [[[[1.0]], [[1.0]]]],
                "fidelity": 1.0,
            },
            {
                "gate": "RX",
                "params": (np.pi / 2.0,),
                "targets": (0,),
                "kraus_ops": [[[[1.0]], [[1.0]]]],
                "fidelity": 1.0,
            },
        ],
        "assignment_probs": {"1": [[1.0, 0.0], [0.0, 1.0]], "0": [[1.0, 0.0], [0.0, 1.0]]},
    }


@pytest.fixture
def qcs_aspen8_device(qcs_aspen8_isa):
    return QCSDevice(quantum_processor_id="Aspen-8", isa=qcs_aspen8_isa)


@pytest.fixture(scope="session")
def qvm(client: Client):
    try:
        qvm = QVMConnection(client=client, random_seed=52)
        qvm.run(Program(I(0)), [])
        return qvm
    except (RequestException, QVMNotRunning, UnknownApiError) as e:
        return pytest.skip("This test requires QVM connection: {}".format(e))
    except QVMVersionMismatch as e:
        return pytest.skip("This test requires a different version of the QVM: {}".format(e))


@pytest.fixture()
def compiler(compiler_device, compiler_isa, client: Client):
    try:
        compiler = QVMCompiler(device=compiler_device, client=client, timeout=1)
        program = Program(I(0))
        compiler.quil_to_native_quil(program)
        return compiler
    except (RequestException, QuilcNotRunning, UnknownApiError, TimeoutError) as e:
        return pytest.skip("This test requires compiler connection: {}".format(e))
    except QuilcVersionMismatch as e:
        return pytest.skip("This test requires a different version of quilc: {}".format(e))


@pytest.fixture()
def dummy_compiler(qcs_aspen8_device: QCSDevice, client: Client):
    return DummyCompiler(qcs_aspen8_device, client)


@pytest.fixture(scope="session")
def client():
    configuration = QCSClientConfiguration.load(
        secrets_file_path=Path(f"{DIR_PATH}/tests/data/qcs_secrets.toml"),
        settings_file_path=Path(f"{DIR_PATH}/tests/data/qcs_settings.toml"),
    )
    return Client(configuration=configuration)


@pytest.fixture(scope="session")
def benchmarker(client: Client):
    try:
        bm = BenchmarkConnection(client=client, timeout=2)
        bm.apply_clifford_to_pauli(Program(I(0)), sX(0))
        return bm
    except (RequestException, TimeoutError) as e:
        return pytest.skip(
            "This test requires a running local benchmarker endpoint (ie quilc): {}".format(e)
        )


def _str_to_bool(s):
    """Convert either of the strings 'True' or 'False' to their Boolean equivalent"""
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError("Please specify either True or False")


def pytest_addoption(parser):
    parser.addoption(
        "--use-seed",
        action="store",
        type=_str_to_bool,
        default=True,
        help="run operator estimation tests faster by using a fixed random seed",
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked as being 'slow'"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture()
def use_seed(pytestconfig):
    return pytestconfig.getoption("use_seed")
