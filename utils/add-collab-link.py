#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "nbformat",
# ]
# ///
import argparse

import nbformat


def main(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
        # _parent means same window, different tab
        collab_link = f"""<a target="_parent" href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/{file_path}">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>"""
        for cell in notebook["cells"]:
            if (
                'href="https://colab.research.google.com/github/gretelai'
                in cell["source"]
            ):
                cell["source"] = collab_link
                break
        else:
            collab_cell = nbformat.v4.new_markdown_cell(collab_link)
            notebook["cells"].insert(0, collab_cell)

    with open(file_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()
    main(args.file_path)
