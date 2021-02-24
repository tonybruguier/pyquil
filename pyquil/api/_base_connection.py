##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
import re
import time
import warnings

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import requests
from httpx import Client, Response
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from pyquil.api._error_reporting import _record_call
from pyquil.api._errors import (
    error_mapping,
    ApiError,
    UserMessageError,
    UnknownApiError,
    TooManyQubitsError,
)

from pyquil.quil import Program
from pyquil.version import __version__
from pyquil.wavefunction import Wavefunction

TYPE_EXPECTATION = "expectation"
TYPE_MULTISHOT = "multishot"
TYPE_MULTISHOT_MEASURE = "multishot-measure"
TYPE_WAVEFUNCTION = "wavefunction"


# def get_json(client: Client, url: str, params: Optional[Dict[Any, Any]] = None) -> Any:
#     """
#     Get JSON from a Forest endpoint.
#     """
#     logger.debug("Sending GET request to %s. Params: %s", url, params)
#     res = client.get(url, params=params)
#     if res.status_code >= 400:
#         raise parse_error(res)
#     return res.json()


def validate_noise_probabilities(noise_parameter: Optional[List[float]]) -> None:
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?

    :param list noise_parameter: List of noise parameter values to be validated.
    """
    if not noise_parameter:
        return
    if not isinstance(noise_parameter, list):
        raise TypeError("noise_parameter must be a list")
    if any([not isinstance(value, float) for value in noise_parameter]):
        raise TypeError("noise_parameter values should all be floats")
    if len(noise_parameter) != 3:
        raise ValueError("noise_parameter lists must be of length 3")
    if sum(noise_parameter) > 1 or sum(noise_parameter) < 0:
        raise ValueError("sum of entries in noise_parameter must be between 0 and 1 (inclusive)")
    if any([value < 0 for value in noise_parameter]):
        raise ValueError("noise_parameter values should all be non-negative")


def validate_qubit_list(qubit_list: Sequence[int]) -> Sequence[int]:
    """
    Check the validity of qubits for the payload.

    :param qubit_list: List of qubits to be validated.
    """
    if not isinstance(qubit_list, Sequence):
        raise TypeError("'qubit_list' must be of type 'Sequence'")
    if any(not isinstance(i, int) or i < 0 for i in qubit_list):
        raise TypeError("'qubit_list' must contain positive integer values")
    return qubit_list


def prepare_register_list(
    register_dict: Dict[str, Union[bool, Sequence[int]]]
) -> Dict[str, Union[bool, Sequence[int]]]:
    """
    Canonicalize classical addresses for the payload and ready MemoryReference instances
    for serialization.

    This function will cast keys that are iterables of int-likes to a list of Python
    ints. This is to support specifying the register offsets as ``range()`` or numpy
    arrays. This mutates ``register_dict``.

    :param register_dict: The classical memory to retrieve. Specified as a dictionary:
        the keys are the names of memory regions, and the values are either (1) a list of
        integers for reading out specific entries in that memory region, or (2) True, for
        reading out the entire memory region.
    """
    if not isinstance(register_dict, dict):
        raise TypeError("register_dict must be a dict but got " + repr(register_dict))

    for k, v in register_dict.items():
        if isinstance(v, bool):
            assert v  # If boolean v must be True
            continue

        indices = [int(x) for x in v]  # support ranges, numpy, ...

        if not all(x >= 0 for x in indices):
            raise TypeError("Negative indices into classical arrays are not allowed.")
        register_dict[k] = indices

    return register_dict


# TODO(andrew): move?
def run_and_measure_payload(
    quil_program: Program, qubits: Sequence[int], trials: int, random_seed: int
) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._run_and_measure`"""
    if not quil_program:
        raise ValueError(
            "You have attempted to run an empty program."
            " Please provide gates or measure instructions to your program."
        )

    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    qubits = validate_qubit_list(qubits)
    if not isinstance(trials, int):
        raise TypeError("trials must be an integer")

    payload = {
        "type": TYPE_MULTISHOT_MEASURE,
        "qubits": list(qubits),
        "trials": trials,
        "compiled-quil": quil_program.out(calibrations=False),
    }

    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


# TODO(andrew): move?
def wavefunction_payload(quil_program: Program, random_seed: int) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._wavefunction`"""
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")

    payload: Dict[str, object] = {
        "type": TYPE_WAVEFUNCTION,
        "compiled-quil": quil_program.out(calibrations=False),
    }

    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


# TODO(andrew): move?
def expectation_payload(
    prep_prog: Program, operator_programs: Optional[Iterable[Program]], random_seed: int
) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._expectation`"""
    if operator_programs is None:
        operator_programs = [Program()]

    if not isinstance(prep_prog, Program):
        raise TypeError("prep_prog variable must be a Quil program object")

    payload: Dict[str, object] = {
        "type": TYPE_EXPECTATION,
        "state-preparation": prep_prog.out(calibrations=False),
        "operators": [x.out(calibrations=False) for x in operator_programs],
    }

    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


# TODO(andrew): move?
def qvm_run_payload(
    quil_program: Program,
    classical_addresses: Dict[str, Union[bool, Sequence[int]]],
    trials: int,
    measurement_noise: Optional[Tuple[float, float, float]],
    gate_noise: Optional[Tuple[float, float, float]],
    random_seed: Optional[int],
) -> Dict[str, object]:
    """REST payload for QVM execution`"""
    if not quil_program:
        raise ValueError(
            "You have attempted to run an empty program."
            " Please provide gates or measure instructions to your program."
        )
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    classical_addresses = prepare_register_list(classical_addresses)
    if not isinstance(trials, int):
        raise TypeError("trials must be an integer")

    payload = {
        "type": TYPE_MULTISHOT,
        "addresses": classical_addresses,
        "trials": trials,
        "compiled-quil": quil_program.out(calibrations=False),
    }

    if measurement_noise is not None:
        payload["measurement-noise"] = measurement_noise
    if gate_noise is not None:
        payload["gate-noise"] = gate_noise
    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload
