#!/usr/bin/env sh

protoc \
  -I=src/agri_vision_edge/third_party/tensorflow_models/research \
  src/agri_vision_edge/third_party/tensorflow_models/research/object_detection/protos/*.proto \
  --python_out=src/agri_vision_edge/third_party/tensorflow_models/research
