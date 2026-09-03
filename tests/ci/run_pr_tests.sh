#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

source_dir="${DEEPEP_SOURCE_DIR:?DEEPEP_SOURCE_DIR is required}"
log_dir="${DEEPEP_TEST_LOG_DIR:?DEEPEP_TEST_LOG_DIR is required}"
num_processes="${DEEPEP_PR_NUM_PROCESSES:-8}"
test_timeout="${DEEPEP_PR_TEST_TIMEOUT_SECONDS:-3600}"

[[ "${num_processes}" =~ ^[1-9][0-9]*$ ]] || {
    echo "DEEPEP_PR_NUM_PROCESSES must be a positive integer" >&2
    exit 2
}
[[ "${test_timeout}" =~ ^[1-9][0-9]*$ ]] || {
    echo "DEEPEP_PR_TEST_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
}

mkdir -p -- "${log_dir}"
available_devices="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
[[ "${available_devices}" =~ ^[0-9]+$ ]] || {
    echo "Unable to determine the visible HCU device count: ${available_devices}" >&2
    exit 1
}
if (( available_devices < num_processes )); then
    echo "PR tests require ${num_processes} HCU devices, but only ${available_devices} are visible" >&2
    exit 1
fi

export MASTER_ADDR=127.0.0.1
export RANK=0
export WORLD_SIZE=1

run_single_node_test() {
    local test_name="$1"
    local master_port="$2"
    local test_file="${source_dir}/tests/${test_name}.py"

    [[ -f "${test_file}" ]] || {
        echo "Required PR test does not exist: ${test_file}" >&2
        return 1
    }

    echo "Running tests/${test_name}.py with ${num_processes} processes on one node"
    MASTER_PORT="${master_port}" \
        timeout --signal=TERM --kill-after=30s "${test_timeout}" \
        python3 -u "${test_file}" --num-processes "${num_processes}" \
        2>&1 | tee "${log_dir}/${test_name}.log"
}

{
    echo "single-node process count: ${num_processes}"
    echo "visible HCU devices: ${available_devices}"
    echo "included: tests/test_intranode.py"
    echo "included: tests/test_low_latency.py"
    echo "excluded: tests/test_internode.py (requires more than one 8-device node)"
} | tee "${log_dir}/single-node-scope.log"

run_single_node_test test_intranode 8361

# Match the heap size used by the repository's existing HCU test environment.
# The low-latency defaults require more than rocSHMEM's 1 GiB default heap.
export ROCSHMEM_HEAP_SIZE=3737418240
run_single_node_test test_low_latency 8362
