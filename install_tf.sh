#!/bin/bash

TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

sudo apt-get install python-pip python-dev vim -y
sudo pip install --upgrade $TF_BINARY_URL
