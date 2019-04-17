#!/bin/bash

DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX1.sh
  echo "Unknown system, assuming DGX1"
fi
DGXNGPU=${PHILLY_GPU_COUNT:-$1}
GPU_PER_NODE=${PHILLY_CONTAINER_GPU_COUNT:-${DGXNGPU}}
JOB_ID=${PHILLY_JOB_ID:-'007'}
NODE_ID=${PHILLY_CONTAINER_INDEX:-'0'}

NUM_NODE=$(expr $DGXNGPU / $GPU_PER_NODE)
MULTI_NODE="--nnodes $NUM_NODE --node_rank $NODE_ID"
echo "Run vars: id $JOB_ID gpus $GPU_PER_NODE mparams $MULTI_NODE"

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
DATASET_DIR=${DATASET_DIR:-'../data'}
RESULTS_DIR='gnmt_wmt16'
BATCH=${BATCH:-32}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
TEST_BATCH_SIZE=32
LR=${LR:-"1.75e-3"}
TARGET=21.80
WARMUP_ITERS=${WARMUP_ITERS:-100}
REMAIN_STEPS=${REMAIN_STEPS:-1450}
DECAY_STEPS=${DECAY_STEPS:-40}

echo "running benchmark"

# run training
python -m torch.distributed.launch --nproc_per_node $GPU_PER_NODE $MULTI_NODE train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --target-bleu $TARGET \
  --epochs 20 \
  --math fp32 \
  --print-freq 500 \
  --batch-size $BATCH \
  --test-batch-size $TEST_BATCH_SIZE \
  --val-batch-size 32 \
  --keep-checkpoints 10 \
  --save-all \
  --model-config "{'num_layers': 8, 'hidden_size': 1024, 'dropout':0.2, 'share_embedding': False}" \
  --optimization-config "{'optimizer': 'FusedAdam', 'lr': $LR}" \
  --scheduler-config "{'lr_method':'mlperf', 'warmup_iters':$WARMUP_ITERS, 'remain_steps':$REMAIN_STEPS, 'decay_steps':$DECAY_STEPS}" ; ret_code=$?
  # --resume ./results/gnmt_wmt16/checkpoint1.pth \
  # --resume ./results/backup/checkpoint5.pth \
  # --start-epoch 2 \

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_TRANSLATOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

