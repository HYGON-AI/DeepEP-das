#!/bin/bash

# rocSHMEM
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config

# # duSHMEM
# export LD_LIBRARY_PATH=/opt/dtk/dushmem/lib:$LD_LIBRARY_PATH
# export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_2:1,1:mlx5_3:1,2:mlx5_4:1,3:mlx5_5:1,4:mlx5_6:1,5:mlx5_7:1,6:mlx5_8:1,7:mlx5_9:1
# export NVSHMEM_SYMMETRIC_SIZE=10737418240

# common
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd)/../

# test
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 ./test_internode.py
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 ./test_internode.py --test-ll-compatibility
torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 ./test_low_latency.py # --pressure-test
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 ./test_low_latency.py --use-logfmt
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 ./test_low_latency.py --enable-dispatch-ll-layered --enable-combine-overlap
