#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

for para in $*
do
    if [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --testmode* ]];then
        testmode=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# default env
DIST_URL=${1}
DIST_PORT=${2}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

# =============================================================================
# 调试输出（确认环境变量传递正确）
# =============================================================================
if [ "$RANK" -eq 0 ]; then
    echo "=== DeepEP Test Start ==="
    echo "Test mode: ${testmode:-internode}"
    echo "World size: $WORLD_SIZE"
    echo "Master: $DIST_URL:$DIST_PORT"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "TEST_DIR: $TEST_DIR"
    echo "ROCSHMEM_TOPO_FILE_FORCE: $ROCSHMEM_TOPO_FILE_FORCE"
    echo "NCCL_PLUGIN: ${NCCL_NET_PLUGIN:-none}"
    echo "NCCL_IB_HCA: ${NCCL_IB_HCA:-auto}"
    echo "HSA_FORCE_FINE_GRAIN_PCIE: ${HSA_FORCE_FINE_GRAIN_PCIE:-not set}"
fi

DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

TEST_BASE_ARGS=(
    --hidden 7168
    --num-experts 256
    --num-topk 8
)

# 三种模式的 APP 定义
case ${testmode} in
    intranode)
        # 节点内测试
        INTRANODE_ARGS=(
            "${TEST_BASE_ARGS[@]}"
            # intranode 特定参数:
            --num-tokens 4096
        )
        APP="python3 -u ${TEST_DIR}/test_intranode.py \
            ${DISTRIBUTED_ARGS[@]} \
            ${INTRANODE_ARGS[@]} \
            "
        ;;
    lowlatency)
        # 低延迟测试
        LOWLATENCY_ARGS=(
            "${TEST_BASE_ARGS[@]}"
            # lowlatency 特定参数:
            --num-tokens 128
            # --pressure-test
        )
        APP="python3 -u ${TEST_DIR}/test_low_latency.py \
            ${DISTRIBUTED_ARGS[@]} \
            ${LOWLATENCY_ARGS[@]} \
            "
        ;;
    internode|*)
        # 跨节点测试（默认）
        INTERNODE_ARGS=(
            "${TEST_BASE_ARGS[@]}"
            # internode 特定参数:
            --num-tokens 4096
            # --test-ll-compatibility
        )
        APP="python3 -u ${TEST_DIR}/test_internode.py \
            ${DISTRIBUTED_ARGS[@]} \
            ${INTERNODE_ARGS[@]} \
            "
        ;;
esac

###############################################################################

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 6 8 32
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_aibenchmark_8nodes_tp4-pp2-ep8-etp2-cp1-vp2
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 6 8 32
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS[@]}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

###############################################################################

echo "launch_with_binding=${launch_with_binding}, APP=${APP}"

#for hygon cpu
${launch_with_binding} ${LOCAL_RANK} ${APP}
