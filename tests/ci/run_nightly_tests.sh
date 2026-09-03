#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

source_dir="${DEEPEP_SOURCE_DIR:?DEEPEP_SOURCE_DIR is required}"
log_dir="${DEEPEP_TEST_LOG_DIR:?DEEPEP_TEST_LOG_DIR is required}"
config_root="${DEEPEP_NIGHTLY_CONFIG_ROOT:?DEEPEP_NIGHTLY_CONFIG_ROOT is required}"
test_profile="${DEEPEP_TEST_PROFILE:?DEEPEP_TEST_PROFILE is required}"
runner_label="${DEEPEP_NIGHTLY_RUNNER_LABEL:?DEEPEP_NIGHTLY_RUNNER_LABEL is required}"
node_rank="${RANK:?RANK is required}"
world_size="${WORLD_SIZE:?WORLD_SIZE is required}"
master_addr="${MASTER_ADDR:?MASTER_ADDR is required}"
master_port="${MASTER_PORT:?MASTER_PORT is required}"
num_processes="${DEEPEP_NIGHTLY_NUM_PROCESSES:-8}"
test_timeout="${DEEPEP_NIGHTLY_TEST_TIMEOUT_SECONDS:-7200}"
peer_timeout="${DEEPEP_NIGHTLY_PEER_TIMEOUT_SECONDS:-1200}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

for value_name in num_processes test_timeout peer_timeout master_port; do
    value="${!value_name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${value_name} must be a positive integer"
done
[[ "${world_size}" == "2" ]] || fail "WORLD_SIZE must be 2 for the fixed two-node test"
case "${runner_label}:${node_rank}" in
    nmz2:0|nmz3:1) ;;
    *) fail "runner label and node rank must be nmz2:0 or nmz3:1" ;;
esac
case "${test_profile}" in
    internode|low-latency|all) ;;
    *) fail "unknown Nightly test profile: ${test_profile}" ;;
esac
(( master_port < 65533 )) || fail "MASTER_PORT must leave room for the preflight and second test ports"

mkdir -p -- "${log_dir}"
node_config="${config_root}/${runner_label}"
runtime_env="${node_config}/runtime.env"
topology_file="${node_config}/topo.config"
[[ -f "${runtime_env}" ]] || fail "runner configuration is missing: ${runtime_env}"
[[ -f "${topology_file}" ]] || fail "runner topology is missing: ${topology_file}"

load_runtime_env() {
    local line key value
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        [[ -z "${line}" || "${line}" == \#* ]] && continue
        [[ "${line}" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.,:/+-]+)$ ]] || \
            fail "invalid runner configuration line in ${runtime_env}"
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        case "${key}" in
            ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX|ROCSHMEM_MAX_NUM_CONTEXTS|\
            ROCSHMEM_ALLOWED_IBV_DEVICES|ROCSHMEM_HEAP_SIZE|\
            ROCSHMEM_DISABLE_HDP_FLUSH|ROCSHMEM_GDR_DISABLE_XDP|\
            ROCSHMEM_IB_GID_INDEX|ROCSHMEM_USE_IB_HCA|\
            UCX_NET_DEVICES|UCX_IB_GID_INDEX|NCCL_IB_HCA|\
            NCCL_SOCKET_IFNAME|NCCL_IB_GID_INDEX|NCCL_IB_ROCE_VERSION_NUM|\
            GLOO_SOCKET_IFNAME)
                export "${key}=${value}"
                ;;
            *) fail "unsupported runner configuration key: ${key}" ;;
        esac
    done < "${runtime_env}"
}

load_runtime_env
for required_name in \
    ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX \
    ROCSHMEM_MAX_NUM_CONTEXTS \
    ROCSHMEM_ALLOWED_IBV_DEVICES \
    ROCSHMEM_HEAP_SIZE; do
    [[ -n "${!required_name:-}" ]] || fail "runner configuration must set ${required_name}"
done
export ROCSHMEM_TOPO_FILE_FORCE="${topology_file}"

available_devices="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
[[ "${available_devices}" =~ ^[0-9]+$ ]] || fail "unable to determine the visible HCU device count"
if (( available_devices < num_processes )); then
    fail "Nightly tests require ${num_processes} HCU devices, but only ${available_devices} are visible"
fi
[[ -d /dev/infiniband ]] || fail "/dev/infiniband is not available inside the test container"
getent ahostsv4 "${master_addr}" >/dev/null || fail "the fixed master host cannot be resolved: ${master_addr}"

IFS=',' read -r -a allowed_devices <<< "${ROCSHMEM_ALLOWED_IBV_DEVICES}"
(( ${#allowed_devices[@]} > 0 )) || fail "ROCSHMEM_ALLOWED_IBV_DEVICES is empty"
for ib_device in "${allowed_devices[@]}"; do
    [[ -d "/sys/class/infiniband/${ib_device}" ]] || \
        fail "configured RDMA device is unavailable on ${runner_label}: ${ib_device}"
done

topology_entries=0
while read -r pci_device ib_device device_index extra; do
    [[ -z "${pci_device}" || "${pci_device}" == \#* ]] && continue
    [[ -z "${extra:-}" ]] || fail "invalid topology entry in ${topology_file}"
    [[ "${device_index}" =~ ^[0-9]+$ ]] || fail "invalid topology device index in ${topology_file}"
    [[ -e "/sys/bus/pci/devices/${pci_device}" ]] || \
        fail "configured HCU PCI device is unavailable on ${runner_label}: ${pci_device}"
    [[ -d "/sys/class/infiniband/${ib_device}" ]] || \
        fail "topology RDMA device is unavailable on ${runner_label}: ${ib_device}"
    (( topology_entries += 1 ))
done < "${topology_file}"
(( topology_entries == num_processes )) || \
    fail "topology must contain ${num_processes} HCU-to-RDMA entries; found ${topology_entries}"

{
    echo "runner label: ${runner_label}"
    echo "runner name: ${RUNNER_NAME:-unknown}"
    echo "node rank: ${node_rank}/${world_size}"
    echo "master host: ${master_addr}"
    echo "visible HCU devices: ${available_devices}"
    echo "configured RDMA devices: ${#allowed_devices[@]}"
    echo "topology entries: ${topology_entries}"
    echo "test profile: ${test_profile}"
} | tee "${log_dir}/internode-preflight.log"

python3 - "${node_rank}" "${master_addr}" "$((master_port + 1))" \
    "${peer_timeout}" "${GITHUB_RUN_ID:-unknown}:${GITHUB_RUN_ATTEMPT:-unknown}" \
    2>&1 <<'PY' | tee -a "${log_dir}/internode-preflight.log"
import socket
import sys
import time

node_rank = int(sys.argv[1])
master_addr = sys.argv[2]
port = int(sys.argv[3])
timeout = int(sys.argv[4])
token = sys.argv[5].encode()

if node_rank == 0:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(1)
        server.settimeout(timeout)
        connection, _ = server.accept()
        with connection:
            if connection.recv(256) != token:
                raise SystemExit("peer preflight token mismatch")
            connection.sendall(b"ok")
else:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((master_addr, port), timeout=5) as connection:
                connection.sendall(token)
                if connection.recv(16) != b"ok":
                    raise RuntimeError("unexpected preflight response")
                break
        except OSError as exc:
            last_error = exc
            time.sleep(2)
    else:
        raise SystemExit(f"peer preflight timed out: {last_error}")

print("two-node TCP preflight: passed", flush=True)
PY

run_two_node_test() {
    local test_name="$1"
    local test_port="$2"
    shift 2
    local test_file="${source_dir}/tests/${test_name}.py"

    [[ -f "${test_file}" ]] || fail "required Nightly test does not exist: ${test_file}"
    echo "Running tests/${test_name}.py on ${runner_label} as node rank ${node_rank}"
    MASTER_PORT="${test_port}" \
        timeout --signal=TERM --kill-after=30s "${test_timeout}" \
        python3 -u "${test_file}" --num-processes "${num_processes}" "$@" \
        2>&1 | tee "${log_dir}/${test_name}.log"
}

case "${test_profile}" in
    internode)
        run_two_node_test test_internode "${master_port}"
        ;;
    low-latency)
        run_two_node_test test_low_latency "$((master_port + 2))"
        ;;
    all)
        run_two_node_test test_internode "${master_port}"
        run_two_node_test test_low_latency "$((master_port + 2))"
        ;;
esac
