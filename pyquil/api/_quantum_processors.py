##############################################################################
# Copyright 2018 Rigetti Computing
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

from pyquil.api import Client
from pyquil.device import QCSDevice
from qcs_api_client.operations.sync import get_instruction_set_architecture


def get_device(client: Client, quantum_processor_id: str) -> QCSDevice:
    isa = client.qcs_request(
        get_instruction_set_architecture, quantum_processor_id=quantum_processor_id
    )

    return QCSDevice(quantum_processor_id=quantum_processor_id, isa=isa)
