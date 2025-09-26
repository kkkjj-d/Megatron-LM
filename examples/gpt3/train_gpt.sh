#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/workspace/datasets/megatron/checkpoint #<Specify path>
TENSORBOARD_LOGS_PATH=Megatron-LM/logs/tb #<Specify path>
VOCAB_FILE=/workspace/datasets/gpt/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/workspace/datasets/gpt/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/share/datasets/pretrain/openwebtext/gpt2/bpe_openwebtext_text_document #<Specify path and file prefix>_text_document
DATA_CACHE_PATH=/workspace/datasets/megatron/cache/openwebtext
LOG_PATH=/workspace/Megatron-LM/logs/($GPUS_PER_NODE)_megatron.log

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 4096
    --num-attention-heads 16
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend unfused
)

TRAINING_ARGS=(
    --micro-batch-size 8
    --global-batch-size 64
    --train-iters 500000
    --init-method-std 0.02
    --clip-grad 1.0
    --lr 6e-4
    --lr-decay-style cosine
    --min-lr 6e-5
    --weight-decay 0.1
    --lr-warmup-fraction 0.01
    --lr-decay-iters 500000
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 9990,9,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee $LOG_PATH
