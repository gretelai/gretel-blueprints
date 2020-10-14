from pathlib import Path
from unittest.mock import patch
import json
import uuid

import pytest

import construct_manifests as cm


@pytest.fixture(autouse=True, scope="module")
def patch_sample_data_map():
    p = patch("construct_manifests.get_gretel_sample_data_map")
    p.start()
    cm.get_gretel_sample_data_map.return_value = {"safecast": "foo"}
    yield
    p.stop()


def default_manifest(name: str):
    return {
        "name": name,
        "description": "this is the description",
        "tags": ["these", "are", "tags"],
        "sample_data_key": "safecast",
        "featured": False,
        "blueprint_url": "https://url",
        "language": "python",
    }


def create_manifest(target_dir: str, manifest: dict, bp_fname: str):
    dest = Path(target_dir / uuid.uuid4().hex)
    dest.mkdir()
    with open(dest / bp_fname, "w") as fout:
        fout.write("# nada")

    with open(dest / "manifest.json", "w") as fout:
        fout.write(json.dumps(manifest))


def test_create_manifest_ok(tmpdir):
    m1 = default_manifest("one")
    create_manifest(tmpdir, m1, "blueprint.py")

    m2 = default_manifest("two")
    m2["featured"] = True
    create_manifest(tmpdir, m2, "blueprint.py")

    check = cm.create_manifest(tmpdir)
    bp = check["blueprint_map"]
    assert len(bp) == 2
    assert "one" in bp
    assert "two" in bp
    assert len(check["featured"]) == 1


def test_create_manifest_bad_blueprint_file(tmpdir):
    m1 = default_manifest("one")
    create_manifest(tmpdir, m1, "blueprint.nope")

    with pytest.raises(cm.ManifestError):
        cm.create_manifest(tmpdir)


def test_empty_dir(tmpdir):
    Path(tmpdir / "empty").mkdir()
    with pytest.raises(cm.ManifestError):
        cm.create_manifest(tmpdir)


def test_duplicate_names(tmpdir):
    m1 = default_manifest("one")
    create_manifest(tmpdir, m1, "blueprint.py")

    m2 = default_manifest("one")
    m2["featured"] = True
    create_manifest(tmpdir, m2, "blueprint.py")

    with pytest.raises(cm.ManifestError):
        cm.create_manifest(tmpdir)


def test_bad_data_sample_name(tmpdir):
    m1 = default_manifest("one")
    m1["sample_data_key"] = "nope"
    create_manifest(tmpdir, m1, "blueprint.py")

    with pytest.raises(cm.ManifestError):
        cm.create_manifest(tmpdir)
