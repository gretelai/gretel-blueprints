import os
from pathlib import Path


def test_sample_data_previews():
    sample_data_dir = "sample_data"
    preview_dir = "sample_data_previews"
    # get list of files in sample_data
    dir = Path(__file__).parent / sample_data_dir
    sample_files = os.listdir(dir)

    for file in sample_files:
        assert (Path(__file__).parent / preview_dir / file).is_file()
