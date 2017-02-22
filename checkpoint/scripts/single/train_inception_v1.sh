#!/bin/bash

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/flowers-models/inception_v3

# Where the dataset is saved to.
DATASET_DIR=/dataset/flowers

EXAMPLE_HOME=../..

# Download the dataset
python ${EXAMPLE_HOME}/download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

python ${EXAMPLE_HOME}/train.py \
  --type=single \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --max_steps=100 \
  --batch_size=32 \
  --save_interval_secs=60 \
  --save_summaries_secs=120 \
  --optimizer=srmsprop \
  --learning_rate=0.01 \
  --weight_decay=0.0004
  #--learning_rate_decay_factor=0.1 \
  #--num_epochs_per_decay=200 \