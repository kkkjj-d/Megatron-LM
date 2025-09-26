#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/workspace/Megatron-LM/checkpoint #<Specify path>
LOG_PATH=/workspace/megatron/Megatron-LM/logs/test.log
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=/workspace/bert-vocab.txt #<Specify path to file>/bert-vocab.json
DATA_PATH=/share/datasets/pretrain/openwebtext/gpt2/bpe_openwebtext_text_document #<Specify path and file prefix>_text_document
DATA_CACHE_PATH=/workspace/datasets/bert
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 512 
    --max-position-embeddings 512 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 32 
    --train-iters 1000000 
    --weight-decay 1e-2 
    --clip-grad 1.0 
    --fp16
    --lr 0.0001
    --lr-decay-iters 990000 
    --lr-decay-style linear 
    --min-lr 1.0e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction .01 
    --clip-grad 1.0 
    --no-gradient-accumulation-fusion
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --data-cache-path $DATA_CACHE_PATH
    --vocab-file $VOCAB_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --eval-iters 10
)
    # --load $CHECKPOINT_PATH 
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py \
    ${BERT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 
    