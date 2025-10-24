export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=rc,rocm
# export ROCSHMEM_UNIQUEID_WITH_MPI=1
export OMPI_MCA_rmaps_base_mapping_policy="slot:numa"
export ROCSHMEM_MAX_NUM_CONTEXTS=32 
export UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ROCSHMEM_HEAP_SIZE=536870912 805306368 10737418240
export ROCSHMEM_HEAP_SIZE=536870912
export PYTHONPATH=${DEEPEP_ROOT:-\$(pwd)}:$PYTHONPATH
torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port=1234 tests/test_internode.py
