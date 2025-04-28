import os
from pathlib import Path

import pytest
import requests
import yaml
from gretel_client import ClientConfig, configure_session
from gretel_client.projects import Project, tmp_project
from gretel_client.projects.models import read_model_config

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


_configs = list((Path(__file__).parent / "config_templates").glob("**/*.yml"))


@pytest.mark.parametrize(
    "_config_file",
    [
        "/".join(str(_config).split("/")[-4:])
        for _config in _configs
        if _config.parent.name != "tuner"
    ],
)
def test_configs(_config_file, project: Project):
    _config_dict = yaml.safe_load(open(_config_file).read())

    if _config_dict.get("version") == 2:
        resp = requests.post(
                f"{_cloud_url}/v2/workflows/validate",
                json=_config_dict,
                headers={"Authorization": _api_key},
            )
    else:
        resp = requests.post(
            f"{_cloud_url}/projects/{project.name}/models",
            json=_config_dict,
            params={"dry_run": "yes"},
            headers={"Authorization": _api_key},
        )
    if resp.status_code != 200:
        print(f"Error for {_cloud_url}, got response: {resp.text}")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "_config_file",
    [
        "/".join(str(_config).split("/")[-4:])
        for _config in _configs
        if _config.parent.name == "tuner"
    ],
)
def test_tuner_configs(_config_file, project: Project):
    tuner_config_dict = yaml.safe_load(open(_config_file).read())
    tuner_config_dict.pop("metric")
    base_config = tuner_config_dict.pop("base_config")
    config = read_model_config(base_config)
    model_config = next(iter(config["models"][0].values()))

    # update the model config with the tuner params
    for section, section_params in tuner_config_dict.items():
        assert section in model_config
        for name, options in section_params.items():
            # tuner param options are always list-like
            value = next(iter(options.values()))[0]
            if name in model_config[section]:
                model_config[section][name] = value
            else:
                model_config[section].setdefault(name, value)

    # execute dry run via the API
    resp = requests.post(
        f"{_cloud_url}/projects/{project.name}/models",
        json=config,
        params={"dry_run": "yes"},
        headers={"Authorization": _api_key},
    )
    if resp.status_code != 200:
        print(f"Error for {_cloud_url}, got response: {resp.text}")
    assert resp.status_code == 200
