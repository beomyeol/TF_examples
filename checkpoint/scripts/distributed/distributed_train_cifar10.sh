#!/bin/bash

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/cifarnet-model

# Where the dataset is saved to.
DATASET_DIR=/dataset/cifar10

EXAMPLE_HOME=../../

PS_HOSTS="172.17.0.2:2222"
WORKER_HOSTS="172.17.0.3:2222"

# Download the dataset
python ${EXAMPLE_HOME}/download_and_convert_data.py \
  --dataset_name=cifar10 \
  --dataset_dir=${DATASET_DIR}

echo "Job: $1, Task id: $2"

python ${EXAMPLE_HOME}/train.py \
  --type=distributed \
  --job_name=$1 \
  --task_id=$2 \
  --ps_hosts=${PS_HOSTS} \
  --worker_hosts=${WORKER_HOSTS} \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --max_steps=100 \
  --batch_size=128 \
  --save_steps=50 \
  --save_summaries_secs=120 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004
  #--save_secs=120 \
  #--preprocessing_name=cifarnet \