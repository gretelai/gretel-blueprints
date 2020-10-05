"""
Construct single manifest files that can be used to look at all possible
Gretel and community Blueprints
"""
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from pathlib import Path
import json
import os

from marshmallow import Schema, fields, validate, ValidationError
import requests
from smart_open import open as smart_open
import botocore
import botocore.session
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig

GRETEL = "gretel"
COMMUNITY = "community"
BLUEPRINT = "blueprint"
FEATURED_MAX = 5
SUPPORTED_BP_FILES = [".py", ".ipynb", ".js", ".go"]
LANGS = ["python", "go", "golang", "javascript"]
GITHUB_URL = "github_url"
NAME = "name"
SAMPLE_DATA_KEY = "sample_data_key"
FEATURED = "featured"
SHIP = "ship-gretel"

REPO_BASE = "https://github.com/gretelai/gretel-blueprints"


client = botocore.session.get_session().create_client("secretsmanager", region_name="us-west-2")  # noqa
cache_config = SecretCacheConfig()
cache = SecretCache(config=cache_config, client=client)
store_data = json.loads(cache.get_secret_string("blueprints/storage"))
gretel_api_key = store_data["gretel"]
stores = (store_data["store_1"], store_data["store_2"])


class ManifestError(Exception):
    pass


class ManifestSchema(Schema):
    name = fields.String(required=True, validate=validate.Length(max=64))
    description = fields.String(required=True, validate=validate.Length(max=160))  # noqa
    tags = fields.List(
        fields.String(validate=validate.Length(max=32)), validate=validate.Length(max=5)  # noqa
    )
    sample_data_key = fields.String(missing=None)
    featured = fields.Boolean(missing=False)
    blueprint_url = fields.String()
    language = fields.String(required=True, validate=validate.OneOf(LANGS))
    blog_url = fields.String(missing=None)


@dataclass
class PrimaryManifest:
    blueprint_map: Dict[str, dict] = field(default_factory=dict)
    """Map all blueprints by name to their individual manifests"""

    featured: List[dict] = field(default_factory=list)
    """A list of the featured blueprints manifests. These manifests will still also
    be in the blueprint map but will be in a list here for easier access
    """


def _check_blueprint_file(manifest_dir: str) -> str:
    for file in Path(manifest_dir).glob("*"):
        if not file.is_file():
            continue

        if file.stem != BLUEPRINT:
            continue

        if file.suffix not in SUPPORTED_BP_FILES:
            raise ManifestError(
                f"Blueprint file is not a supported blueprint type: {file.name} in {manifest_dir}"  # noqa
            )  # noqa

        return file.name

    raise ManifestError(f"Blueprint file missing from {manifest_dir}")


def process_manifest_dir(manifest_dir: str, subdir: str, sample_data_map: dict) -> dict:  # noqa
    _base = Path(manifest_dir)

    manifest_file = _base / "manifest.json"
    if not manifest_file.is_file():
        raise ManifestError(f"Directory {manifest_dir} missing manifest.json!")

    try:
        manifest_dict = json.loads(open(manifest_file).read())
    except json.JSONDecodeError as err:
        raise ManifestError(
            f"Manifest file in {manifest_dir} not JSON"
        ) from err  # noqa

    try:
        manifest_dict = ManifestSchema().load(manifest_dict)
    except ValidationError as err:
        raise ManifestError(f"Invalid manifest in {manifest_dir}") from err

    blueprint_file = _check_blueprint_file(manifest_dir)

    manifest_dict["github_url"] = f"{REPO_BASE}/{subdir}/{_base.name}/{blueprint_file}"  # noqa

    sample_data = manifest_dict.get("sample_data_key")
    if sample_data and sample_data not in sample_data_map:
        raise ManifestError(
            f"Invalid sample data key: {sample_data} in {manifest_dir}"
        )  # noqa

    return manifest_dict


def get_gretel_sample_data_map() -> dict:
    headers = {"Authorization": gretel_api_key}
    return requests.get(
        "https://api.gretel.cloud/records/samples", headers=headers
    ).json()["data"]["samples"]


def create_manifest(base_dir: str) -> dict:
    gretel_sample_data = get_gretel_sample_data_map()
    manifest = PrimaryManifest()
    for subdir in Path(base_dir).glob("*"):
        if not subdir.is_dir():
            continue
        manifest_dict = process_manifest_dir(
            str(subdir.resolve()), base_dir, gretel_sample_data
        )  # noqa

        if subdir.name.startswith("_"):
            continue

        if manifest_dict[NAME] in manifest.blueprint_map:
            raise ManifestError(
                f"The blueprint name {manifest_dict[NAME]} is already in use!"
            )  # noqa
        manifest.blueprint_map[manifest_dict[NAME]] = manifest_dict
        if manifest_dict[FEATURED]:
            manifest.featured.append(manifest_dict)
        if len(manifest.featured) > FEATURED_MAX:
            raise ManifestError(
                "The maximum amount of featured blueprints has been exceeded"
            )

    return asdict(manifest)


def deploy_manifest(manifest: dict, deploy_mode: str, manifest_type: str):
    if deploy_mode == SHIP:
        for store in stores:
            dest = store + manifest_type + ".json"
            with smart_open(dest, "w") as fout:
                fout.write(json.dumps(manifest))


if __name__ == "__main__":
    try:
        deploy_mode = sys.argv[1]
    except IndexError:
        deploy_mode = None

    for base in (GRETEL,):
        manifest_dict = create_manifest(base)

        if deploy_mode in (SHIP,):
            deploy_manifest(manifest_dict, deploy_mode, base)
