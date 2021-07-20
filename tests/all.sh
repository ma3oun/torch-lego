#!/usr/bin/env bash
export PYTHONPATH="$(pwd)/../src:$PYTHONPATH"
python convNet.py convNet.yaml cuda
python dcgan.py dcgan.yaml cuda