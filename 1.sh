pgrep -f /usr/bin/python | xargs kill -9

export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=rc,rocm
# export ROCSHMEM_UNIQUEID_WITH_MPI=1
export OMPI_MCA_rmaps_base_mapping_policy="slot:numa"
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ROCSHMEM_HEAP_SIZE=536870912 805306368 10737418240
export ROCSHMEM_HEAP_SIZE=10737418240
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 tests/test_internode.py
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 tests/test_low_latency.py
# torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 tests/test_low_latency_new.py --pressure-test
torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 tests/test_internode.py --test-ll-compatibility
