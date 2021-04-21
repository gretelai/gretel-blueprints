import os
from pathlib import Path

import pytest
import requests
from gretel_client import get_cloud_client
import yaml


_api_key = os.getenv("GRETEL_API_KEY")
if not _api_key:
    raise RuntimeError("API key env not set")


@pytest.fixture(scope="module")
def project_name():
    client = get_cloud_client("api-dev", _api_key)
    proj = client.get_project(create=True)
    yield proj.name
    proj.delete()


_configs = (Path(__file__).parent / "config_templates").glob("**/*.yml")


@pytest.mark.parametrize("config_file", _configs)
def test_configs(config_file, project_name):
    _config_file = str(config_file)
    _config_dict = yaml.safe_load(open(_config_file).read())
    resp = requests.post(
        f"https://api-dev.gretel.cloud/projects/{project_name}/models",
        json=_config_dict,
        params={"dry_run": "yes"},
        headers={"Authorization": _api_key}
    )
    assert resp.status_code == 200
