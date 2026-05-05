#!/usr/bin/env sh
# This installs a working combination of packages, although the environment is inconsistent.

pip install -q \
  numpy==1.23.5 \
  scipy==1.10.1 \
  tf_slim \
  pycocotools \
  lvis \
  Cython \
  contextlib2 \
  pillow \
  matplotlib \
  gin-config

pip install -q \
  tf-models-official==2.11.0 --no-deps
