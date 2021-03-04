import networkx as nx
import numpy as np
import pytest
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials

from rpcq.messages import ParameterAref

from pyquil.parser import parse
from pyquil import Program, get_qc
from pyquil.api import QuantumComputer, QPU, QPUCompiler, Client
from pyquil.device import NxDevice, Device
from pyquil.gates import I, X
from pyquil.quilatom import Expression


# TODO(andrew): address these tests

def test_qpu_run(client: Client):
    g = nx.Graph()
    g.add_node(0)
    device = NxDevice(g)

    qc = QuantumComputer(
        name="pyQuil test QC",
        qam=QPU(processor_id="test", client=client),
        device=device,
        compiler=QPUCompiler(processor_id="test", device=device, client=client),
    )
    bitstrings = qc.run_and_measure(program=Program(X(0)), trials=1000)
    assert bitstrings[0].shape == (1000,)
    assert np.mean(bitstrings[0]) > 0.8
    bitstrings = qc.run(qc.compile(Program(X(0))))
    assert bitstrings.shape == (0, 0)


GATE_ARITHMETIC_PROGRAMS = [
    Program(
        """
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[3]
RX(pi/2) 0
RZ(3*theta) 0
RZ(beta+theta) 0
RX(-pi/2) 0
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""
    ),
    Program(
        """
RESET
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[2]
RX(pi/2) 0
RZ(theta) 0
    """
    ),
    Program(
        """
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[3]
RX(pi/2) 0
RZ(0.79*theta) 0
RZ(2*beta+theta*0.5+beta+beta) 0
RX(-pi/2) 0
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""
    ),
    Program(
        """
RX(pi) 0
"""
    ),
]


@pytest.fixture
def processor_id():
    return "U057-W1845B-088-B1-a2-ISW"


@pytest.fixture
def mock_qpu(processor_id):
    return QPU(processor_id=processor_id)


@pytest.fixture
def qpu_compiler(test_device: Device, client: Client):
    compiler = QPUCompiler(
        processor_id=processor_id,
        device=test_device,
        client=client,
        timeout=0.5,
    )
    compiler.quil_to_native_quil(Program(I(0)))
    return compiler


@pytest.fixture
def gate_arithmetic_binaries(qpu_compiler: QPUCompiler):
    return [qpu_compiler.native_quil_to_executable(p) for p in GATE_ARITHMETIC_PROGRAMS]


def test_load(gate_arithmetic_binaries, mock_qpu):
    def test_binary(binary):
        assert hasattr(binary, "recalculation_table")
        mock_qpu.load(binary)
        assert mock_qpu.status == "loaded"
        for mref, rule in mock_qpu._executable.recalculation_table.items():
            assert isinstance(mref, ParameterAref)
            assert isinstance(rule, Expression)
        assert len(mock_qpu._executable.recalculation_table) in [0, 2]

    for bin in gate_arithmetic_binaries:
        test_binary(bin)


def test_build_patch_tables(gate_arithmetic_binaries, mock_qpu):
    for idx, bin in enumerate(gate_arithmetic_binaries[:-1]):
        mock_qpu.load(bin)
        theta = np.random.randint(-100, 100) + np.random.random()
        beta = np.random.randint(-100, 100) + np.random.random()
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu.write_memory(region_name="beta", value=beta)
        patch_table = mock_qpu._build_patch_values()
        assert "theta" in patch_table.keys()
        assert "beta" in patch_table.keys()
        if idx == 0 or idx == 2:
            assert len(patch_table) == 3
        for parameter_name, values in patch_table.items():
            assert isinstance(parameter_name, str)
            assert isinstance(values, list)
            for v in values:
                assert isinstance(v, float) or isinstance(v, int)
            if (idx == 0 or idx == 2) and parameter_name not in ("theta", "beta"):
                assert len(values) == 2


def test_recalculation(gate_arithmetic_binaries, mock_qpu):
    bin = gate_arithmetic_binaries[0]
    mock_qpu.load(bin)
    for theta in np.linspace(0, 1, 50):
        beta = -1 * np.random.random()
        mock_qpu.write_memory(region_name="beta", value=beta)
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu._update_variables_shim_with_recalculation_table()
        assert any(np.isclose(v, 3 * theta) for v in mock_qpu._variables_shim.values())
        assert any(np.isclose(v, theta + beta) for v in mock_qpu._variables_shim.values())
        assert any(np.isclose(v, theta) for v in mock_qpu._variables_shim.values())
    bin = gate_arithmetic_binaries[2]
    mock_qpu.load(bin)
    beta = np.random.random()
    mock_qpu.write_memory(region_name="beta", value=beta)
    for theta in np.linspace(0, 1, 10):
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu._update_variables_shim_with_recalculation_table()
        assert any(np.isclose(v, 4 * beta + 0.5 * theta) for v in mock_qpu._variables_shim.values())


def test_resolve_mem_references(gate_arithmetic_binaries, mock_qpu):
    def expression_test(expression, expected_val):
        expression = parse_expression(expression)
        assert np.isclose(mock_qpu._resolve_memory_references(expression), expected_val)

    def test_theta_and_beta(theta, beta):
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu.write_memory(region_name="beta", value=beta)
        expression_test("SQRT(2) + theta", np.sqrt(2) + theta)
        expression_test("beta*2 + 1", beta * 2 + 1)
        expression_test("(beta + 2) * (1 + theta)", (beta + 2) * (1 + theta))
        expression_test("COS(beta)*SIN(theta)", np.cos(beta) * np.sin(theta))
        expression_test("beta * theta", beta * theta)
        expression_test("theta - beta", theta - beta)

    # We just need the status to be loaded so we can write memory
    mock_qpu.load(gate_arithmetic_binaries[0])
    test_theta_and_beta(0.4, 3.1)
    test_theta_and_beta(5, 0)
    for _ in range(10):
        test_theta_and_beta(np.random.random(), np.random.random() + np.random.randint(-100, 100))


def parse_expression(expression):
    """ We have to use this as a hack for now, RZ is meaningless. """
    return parse(f"RZ({expression}) 0")[0].params[0]


def test_run_expects_executable(mock_qpu: QPU):
    # https://github.com/rigetti/pyquil/issues/740
    qc = get_qc("1q-qvm")
    qc.qam = mock_qpu

    p = Program(X(0))
    with pytest.raises(TypeError, match="It looks like you have provided a Program where an executable is expected"):
        qc.run(p)
