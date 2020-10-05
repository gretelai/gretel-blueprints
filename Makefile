SHELL:=/bin/bash

manifest-check:
	python construct_manifests.py

manifest-ship-s3:
	python construct_manifests.py ship-s3