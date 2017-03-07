#!/bin/bash

MAX_STEPS=1000

PS_HOSTS=172.17.0.2:2222,172.17.0.3:2222
WORKER_HOSTS=172.17.0.1:2222

USAGE="$0 [JOB NAME: ps or worker] [ID]"

if [ "$1" == "" ]; then
  echo "ERROR: No job name"
  echo $USAGE
  exit
fi

if [ ! "$1" == "ps" ] && [ ! "$1" == "worker" ]; then
  echo "ERROR: Invalid job name: $1"
  echo $USAGE
  exit
fi

re='^[0-9]+$'
if ! [[ $2 =~ $re ]]; then
  echo "ERROR: Invalid id: $2"
  echo $USAGE
  exit
fi

python trainer_v1.0.py \
  --ps_hosts=${PS_HOSTS} \
  --worker_hosts=${WORKER_HOSTS} \
  --job_name=$1 \
  --task_index=$2 \
  --max_steps=${MAX_STEPS}
