#!/bin/bash

docker run -d   --name spatiallm-api   --gpus all   -p 8000:8000 spatiallm-api -v ./models:/workspace/SpatialLM/models