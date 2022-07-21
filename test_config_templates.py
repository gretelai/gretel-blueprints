import os
from pathlib import Path

import pytest
import requests
import yaml
from gretel_client import ClientConfig, configure_session
from gretel_client.projects import Project, tmp_project

_api_key = os.getenv("GRETEL_API_KEY")
if not _api_key:
    raise RuntimeError("API key env not set")

_cloud_url = os.getenv("GRETEL_CLOUD_URL")
if not _cloud_url:
    raise RuntimeError("Gretel cloud url env not set")

configure_session(ClientConfig(endpoint=_cloud_url, api_key=_api_key))


@pytest.fixture(scope="session")
def project() -> Project:
    with tmp_project() as project:
        yield project


_configs = (Path(__file__).parent / "config_templates").glob("**/*.yml")


@pytest.mark.parametrize(
    "_config_file", ["/".join(str(_config).split("/")[-4:]) for _config in _configs]
)
def test_configs(_config_file, project: Project):
    _config_dict = yaml.safe_load(open(_config_file).read())
    resp = requests.post(
        f"{_cloud_url}/projects/{project.name}/models",
        json=_config_dict,
        params={"dry_run": "yes"},
        headers={"Authorization": _api_key},
    )
    assert resp.status_code == 200
