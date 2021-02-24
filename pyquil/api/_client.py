import re
from contextlib import contextmanager
from json.decoder import JSONDecodeError
from typing import Optional, Any

import httpx
from qcs_api_client.client import QCSClientConfiguration, build_sync_client

from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
from pyquil.api._logger import logger


class Client:
    """
    Class for interacting with network resources.
    """

    _http: Optional[httpx.Client]
    _config: QCSClientConfiguration

    def __init__(self, http: Optional[httpx.Client] = None, configuration: Optional[QCSClientConfiguration] = None):
        """
        Instantiate a client.

        :param http: optional underlying HTTP client. If none is provided, a default client will be created using
               ``configuration``.
        :param configuration: QCS configuration.
        """
        self._config = configuration or QCSClientConfiguration.load()
        self._http = http

    def post_json(self, url: str, json: Any) -> httpx.Response:
        """
        Post JSON to a URL. Will raise an exception for response statuses >= 400.

        :param url: URL to post to.
        :param json: JSON body of request.
        :return: HTTP response corresponding to request.
        """
        logger.debug("Sending POST request to %s. Body: %s", url, json)
        with self._client() as client:
            res = client.post(url, json=json)
            if res.status_code >= 400:
                raise _parse_error(res)
        return res

    @property
    def qvm_url(self) -> str:
        """
        QVM URL from client configuration.
        """
        return self._config.profile.applications.pyquil.qvm_url

    @property
    def quilc_url(self) -> str:
        """
        Quil compiler URL from client configuration.
        """
        return self._config.profile.applications.pyquil.quilc_url

    def qvm_version(self) -> str:
        """
        Get QVM version string.
        """
        response = self.post_json(self.qvm_url, {"type": "version"})
        split_version_string = response.text.split()
        try:
            qvm_version = split_version_string[0]
        except ValueError:
            raise TypeError(f"Malformed version string returned by the QVM: {response.text}")
        return qvm_version

    @contextmanager
    def _client(self) -> httpx.Client:
        if self._http is None:
            with build_sync_client(configuration=self._config) as client:
                yield client
        else:
            yield self._http


def _parse_error(res: httpx.Response) -> ApiError:
    """
    Errors should contain a "status" field with a human readable explanation of
    what went wrong as well as a "error_type" field indicating the kind of error that can be mapped
    to a Python type.

    There's a fallback error UnknownError for other types of exceptions (network issues, api
    gateway problems, etc.)
    """
    try:
        body = res.json()
    except JSONDecodeError:
        raise UnknownApiError(res.text)

    if "error_type" not in body:
        raise UnknownApiError(str(body))

    error_type = body["error_type"]
    status = body["status"]

    if re.search(r"[0-9]+ qubits were requested, but the QVM is limited to [0-9]+ qubits.", status):
        return TooManyQubitsError(status)

    error_cls = error_mapping.get(error_type, UnknownApiError)
    return error_cls(status)