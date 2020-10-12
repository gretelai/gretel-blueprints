SHELL:=/bin/bash

manifest-check:
	python construct_manifests.py

manifest-ship-gretel:
	python construct_manifests.py ship-gretel
