#!/bin/bash

CUDNN_TAR_NAME="cudnn-8.0-linux-x64-v5.1.tar.gz"

if [ ! -d "cuda" ]; then
if [ ! -d "${CUDNN_TAR_NAME}" ]; then
  echo "Failed to find cudnn tar.gz file"
  exit 1
else
  tar xzvf ${CUDNN_TAR_NAME}
fi
fi

sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
