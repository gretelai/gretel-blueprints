from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from gretel_client import configure_session
from gretel_client.config import GretelClientConfigurationError, get_session_config
from gretel_client.users.users import get_me


class GretelValidationError(Exception):
    pass


def validate_gretel() -> Optional[str]:
    try:
        configure_session(validate=True)
    except GretelClientConfigurationError:
        raise GretelValidationError(
            "Could not authenticate to Gretel. Please verify your API key is configured."
        )

    user_dict = get_me()
    return user_dict["email"]


MAX_SIZE = 50


@dataclass
class TabLLMRequest:
    model_id: str
    prompt: str
    temperature: float
    top_k: int
    top_p: float


class TabLLMStreamError(Exception):
    pass


class TabLLMStream:
    request: TabLLMRequest
    target_count: int
    generated_count: int = 0
    stream_meta_id: str

    _curr_stream_id: Optional[str] = None
    _next_iter: Optional[str] = None

    def __init__(self, *, request: TabLLMRequest, record_count: int):
        self.request = request
        self.target_count = record_count
        self.stream_meta_id = uuid.uuid4().hex

    def _reset_stream(self) -> None:
        self._curr_stream_id = None
        self._next_iter = None

    def _create_stream(self) -> None:
        """
        Create a new TabLLM stream via the Gretel API
        """
        payload = {
            "model_id": self.request.model_id,
            "num_rows": min(MAX_SIZE, self.target_count - self.generated_count),
            "params": {
                "temperature": self.request.temperature,
                "top_k": self.request.top_k,
                "top_p": self.request.top_p,
            },
            "prompt": self.request.prompt,
        }

        print("Creating Gretel TabLLM inference stream.")
        resp = _do_api_call("post", "/v1/inference/tabular/stream", body=payload)
        self._curr_stream_id = resp.get("stream_id")
        self._next_iter = None
        print(f"Created Gretel TabLLM inference stream: {self._curr_stream_id}")

    def iter_stream(self) -> Iterator[dict[str, Any]]:
        while self.generated_count < self.target_count:
            # Make an API request for a new Gretel stream if need be
            if not self._curr_stream_id:
                self._create_stream()

            # Poll the stream for new records
            payload = {
                "count": MAX_SIZE,
                "iterator": self._next_iter,
                "stream_id": self._curr_stream_id,
            }
            resp = _do_api_call(
                "post", "/v1/inference/tabular/stream/iterate", body=payload
            )
            # Extra defensive incase the data key is totally missing.
            if (data_list := resp.get("data")) is None:
                continue
            for record in data_list:
                data_type = record["data_type"]
                if data_type == "TabularResponse":
                    row_data = record["data"]["table_data"]
                    for row in row_data:
                        self.generated_count += 1
                        yield row
                    if self.generated_count >= self.target_count:
                        break
                elif data_type == "logger.error":
                    raise TabLLMStreamError(record["data"])

            # The stream is exhausted when both the state is closed and there
            # is no more data.
            if resp.get("stream_state", {}).get("status") == "closed" and not data_list:
                self._curr_stream_id = None
            else:
                self._next_iter = resp.get("next_iterator")

            time.sleep(0.5)


def get_tabllm_models() -> dict[str, str]:
    """
    Return all available models as a mapping of model
    """
    resp = _do_api_call("get", "/v1/inference/models")
    model_list = resp.get("models", [])
    return {model.get("model_id"): model.get("model_name") for model in model_list}


def _do_api_call(
    method: str,
    path: str,
    query_params: Optional[dict] = None,
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Make a direct API call to Gretel Cloud.

    Args:
        method: "get", "post", etc
        path: The full path to make the request to, any path params must be already included.
            Example: "/users/me"
        query_params: Optional URL based query parameters
        body: An optional JSON payload to send
        headers: Any custom headers that need to bet set.

    NOTE:
        This function will automatically inject the appropriate API hostname and
        authentication from the Gretel configuration.
    """
    if headers is None:
        headers = {}

    method = method.upper()

    if not path.startswith("/"):
        path = "/" + path

    api = get_session_config()._get_api_client()

    # Utilize the ApiClient method to inject the proper authentication
    # into our headers, since Gretel only uses header-based auth we don't
    # need to pass any other data into this
    #
    # NOTE: This function does a pointer-like update of ``headers``
    api.update_params_for_auth(
        headers, None, api.configuration.auth_settings(), None, None, None
    )

    url = api.configuration.host + path

    response = api.request(
        method, url, query_params=query_params, body=body, headers=headers
    )

    return json.loads(response.data.decode())
