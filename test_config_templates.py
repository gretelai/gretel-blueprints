import os
from pathlib import Path

import pytest
import requests
import yaml
from gretel_client import ClientConfig, configure_session
from gretel_client.projects import tmp_project

_api_key = os.getenv("GRETEL_API_KEY")
if not _api_key:
    raise RuntimeError("API key env not set")


configure_session(
    ClientConfig(endpoint="https://api-dev.gretel.cloud", api_key=_api_key)
)


_configs = (Path(__file__).parent / "config_templates").glob("**/*.yml")


@pytest.mark.parametrize(
    "_config_file", ["/".join(str(_config).split("/")[-4:]) for _config in _configs]
)
def test_configs(_config_file):
    with tmp_project() as project:
        _config_dict = yaml.safe_load(open(_config_file).read())
        resp = requests.post(
            f"https://api-dev.gretel.cloud/projects/{project.name}/models",
            json=_config_dict,
            params={"dry_run": "yes"},
            headers={"Authorization": _api_key},
        )
        assert resp.status_code == 200
