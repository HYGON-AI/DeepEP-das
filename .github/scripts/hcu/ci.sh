#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

readonly ROCSHMEM_PATH="third-party/rocshmem"
readonly ROCSHMEM_URL="https://github.com/HYGON-AI/rocSHMEM-das.git"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

resolve_dir() {
    local path="$1"
    [[ -d "${path}" ]] || die "directory does not exist: ${path}"
    realpath "${path}"
}

append_summary() {
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        printf '%s\n' "$*" >> "${GITHUB_STEP_SUMMARY}"
    fi
}

restore_workspace() {
    local workspace runner_temp work_root owner
    workspace="$(realpath -m -- "$1")"
    runner_temp="$(realpath -m -- "${RUNNER_TEMP:?RUNNER_TEMP is required}")"
    work_root="$(dirname -- "${runner_temp}")"

    case "${workspace}" in
        "${work_root}"/*/*) ;;
        *) die "refusing to change ownership outside ${work_root}: ${workspace}" ;;
    esac

    owner="$(stat -c '%u:%g' -- "${runner_temp}")"
    [[ "${owner}" =~ ^[1-9][0-9]*:[0-9]+$ ]] || die "unsafe runner ownership: ${owner}"
    chown -R -- "${owner}" "${workspace}"
}

validate_pr_merge() {
    local source_dir base_sha head_sha parents
    source_dir="$(resolve_dir "$1")"
    base_sha="$2"
    head_sha="$3"

    [[ "${base_sha}" =~ ^[0-9a-f]{40}$ ]] || die "invalid base SHA: ${base_sha}"
    [[ "${head_sha}" =~ ^[0-9a-f]{40}$ ]] || die "invalid head SHA: ${head_sha}"
    read -r -a parents <<< "$(git -C "${source_dir}" show -s --format=%P HEAD)"
    [[ "${#parents[@]}" -eq 2 ]] || die "PR merge ref must have exactly two parents"
    [[ "${parents[0]}" == "${base_sha}" ]] || die "PR merge base does not match the event base SHA"
    [[ "${parents[1]}" == "${head_sha}" ]] || die "PR merge head does not match the event head SHA"

    append_summary "- PR merge ref: verified against base and head SHAs"
}

checkout_rocshmem() {
    local source_dir configured_url gitlink_sha token auth_header
    source_dir="$(resolve_dir "$1")"
    token="${HYGON_AI_CI_TOKEN:?HYGON_AI_CI_TOKEN is required}"

    configured_url="$(git -C "${source_dir}" config -f .gitmodules --get "submodule.${ROCSHMEM_PATH}.url")"
    [[ "${configured_url}" == "${ROCSHMEM_URL}" ]] || die "unexpected rocSHMEM submodule URL: ${configured_url}"

    gitlink_sha="$(git -C "${source_dir}" ls-tree HEAD "${ROCSHMEM_PATH}" | awk '{print $3}')"
    [[ "${gitlink_sha}" =~ ^[0-9a-f]{40}$ ]] || die "invalid rocSHMEM gitlink SHA: ${gitlink_sha}"

    auth_header="$(printf 'x-access-token:%s' "${token}" | base64 | tr -d '\r\n')"
    git -C "${source_dir}" \
        -c "http.https://github.com/.extraheader=Authorization: Basic ${auth_header}" \
        submodule update --init --recursive -- "${ROCSHMEM_PATH}"

    [[ "$(git -C "${source_dir}/${ROCSHMEM_PATH}" rev-parse HEAD)" == "${gitlink_sha}" ]] || \
        die "rocSHMEM checkout does not match the recorded gitlink"
    unset token auth_header

    append_summary "- rocSHMEM submodule: verified and checked out at the recorded gitlink"
}

install_dtk() {
    local package_source package_name archive_path top_dir
    package_source="$1"
    package_name="$(basename -- "${package_source}")"
    archive_path="/opt/${package_name}"

    if [[ "${package_source}" == http://* || "${package_source}" == https://* ]]; then
        curl --fail --location --output "${archive_path}" "${package_source}"
    else
        [[ -f "${package_source}" ]] || die "DTK package does not exist: ${package_source}"
        cp -f -- "${package_source}" "${archive_path}"
    fi

    top_dir="$(tar -tzf "${archive_path}" | awk 'NR == 1 { split($0, parts, "/"); print parts[1] }')"
    [[ "${top_dir}" =~ ^dtk-[A-Za-z0-9._-]+$ ]] || die "unexpected DTK archive root: ${top_dir}"
    rm -rf -- "/opt/${top_dir}" /opt/dtk
    tar -xzf "${archive_path}" -C /opt
    mv -- "/opt/${top_dir}" /opt/dtk
    [[ -f /opt/dtk/env.sh ]] || die "DTK environment file is missing"
}

source_dtk() {
    set +u
    # shellcheck disable=SC1091
    source /opt/dtk/env.sh
    set -u
}

install_build_dependencies() {
    local torch_version="$1"
    python3 -m pip install --no-cache-dir \
        "torch==${torch_version}" \
        pyyaml \
        hypothesis \
        ninja \
        monkeytype \
        wheel \
        setuptools \
        ciupload
}

build_wheel() {
    local source_dir build_variant torch_version dtk_package output_dir dtk_version git_commit
    local -a raw_wheels build_args repaired_wheels
    source_dir="$(resolve_dir "$1")"
    build_variant="$2"
    torch_version="$3"
    dtk_package="$4"
    output_dir="$(realpath -m -- "$5")"

    case "${build_variant}" in
        standard) build_args=(rocshmem) ;;
        shca) build_args=(rocshmem BUILD_SHCA=ON) ;;
        *) die "unknown build variant: ${build_variant}" ;;
    esac

    case "${output_dir}" in
        "${source_dir}"/*) ;;
        *) die "output directory must be inside the source checkout: ${output_dir}" ;;
    esac

    install_dtk "${dtk_package}"
    source_dtk
    install_build_dependencies "${torch_version}"

    rm -rf -- "${source_dir}/build_" "${source_dir}/dist" "${output_dir}"
    mkdir -p -- "${output_dir}"
    (
        cd "${source_dir}"
        bash build.sh "${build_args[@]}"
    )

    shopt -s nullglob
    raw_wheels=("${source_dir}"/dist/*.whl)
    shopt -u nullglob
    (( ${#raw_wheels[@]} > 0 )) || die "build did not produce a wheel"

    if [[ "${dtk_package}" =~ dtk([0-9]+)\.([0-9]+) ]]; then
        dtk_version="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    else
        die "cannot derive the DTK version from ${dtk_package}"
    fi
    git_commit="$(git -C "${source_dir}" rev-parse --short=6 HEAD)"

    CIUpload REPAIR \
        --dtk_version "${dtk_version}" \
        --torch_version "${torch_version}" \
        --git_commit "${git_commit}" \
        --outputdir "${output_dir}" \
        -f "${raw_wheels[@]}"

    shopt -s nullglob
    repaired_wheels=("${output_dir}"/*.whl)
    shopt -u nullglob
    (( ${#repaired_wheels[@]} > 0 )) || die "CIUpload did not produce a repaired wheel"

    python3 -m pip install --no-deps --force-reinstall "${repaired_wheels[@]}"
    (
        cd /tmp
        python3 - <<'PY'
import deep_ep

print(f"deep_ep loaded from: {deep_ep.__file__}")
print(f"deep_ep version: {getattr(deep_ep, '__version__', 'unknown')}")
PY
    )

    append_summary "- build variant: ${build_variant}"
    append_summary "- Torch version: ${torch_version}"
    append_summary "- repaired wheels: ${#repaired_wheels[@]}"
    printf '%s\n' "${repaired_wheels[@]}"
}

run_pr_tests() {
    local source_dir wheel_dir log_dir hook
    source_dir="$(resolve_dir "$1")"
    wheel_dir="$(resolve_dir "$2")"
    log_dir="$(realpath -m -- "$3")"
    hook="${source_dir}/tests/ci/run_pr_tests.sh"
    mkdir -p -- "${log_dir}"

    export DEEPEP_SOURCE_DIR="${source_dir}"
    export DEEPEP_WHEEL_DIR="${wheel_dir}"
    export DEEPEP_TEST_LOG_DIR="${log_dir}"

    if [[ -f "${hook}" ]]; then
        bash "${hook}"
        append_summary "- PR unit-test hook: passed"
    else
        echo "PR unit-test hook is not configured: tests/ci/run_pr_tests.sh" | tee "${log_dir}/ut-hook-status.log"
        append_summary "- PR unit-test hook: not configured; build and import checks only"
    fi
}

run_nightly_tests() {
    local source_dir wheel_dir log_dir hook
    source_dir="$(resolve_dir "$1")"
    wheel_dir="$(resolve_dir "$2")"
    log_dir="$(realpath -m -- "$3")"
    hook="${source_dir}/tests/ci/run_nightly_tests.sh"
    mkdir -p -- "${log_dir}"

    [[ -n "${DEEPEP_TEST_PROFILE:-}" ]] || die "DEEPEP_TEST_PROFILE is required"
    [[ -n "${DEEPEP_NIGHTLY_CONFIG_ROOT:-}" ]] || die "DEEPEP_NIGHTLY_CONFIG_ROOT is required"
    [[ -d "${DEEPEP_NIGHTLY_CONFIG_ROOT}" ]] || die "Nightly configuration root does not exist: ${DEEPEP_NIGHTLY_CONFIG_ROOT}"
    [[ -f "${hook}" ]] || die "Nightly test hook is not configured: tests/ci/run_nightly_tests.sh"

    export DEEPEP_SOURCE_DIR="${source_dir}"
    export DEEPEP_WHEEL_DIR="${wheel_dir}"
    export DEEPEP_TEST_LOG_DIR="${log_dir}"
    bash "${hook}"
    append_summary "- Nightly test profile: ${DEEPEP_TEST_PROFILE}"
    append_summary "- Nightly test hook: passed"
}

usage() {
    cat <<'EOF'
Usage:
  ci.sh restore-workspace <workspace>
  ci.sh validate-pr-merge <source-dir> <base-sha> <head-sha>
  ci.sh checkout-rocshmem <source-dir>
  ci.sh build-wheel <source-dir> <standard|shca> <torch-version> <dtk-package> <output-dir>
  ci.sh run-pr-tests <source-dir> <wheel-dir> <log-dir>
  ci.sh run-nightly-tests <source-dir> <wheel-dir> <log-dir>
EOF
}

command="${1:-}"
case "${command}" in
    restore-workspace)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        restore_workspace "$2"
        ;;
    validate-pr-merge)
        [[ "$#" -eq 4 ]] || { usage; exit 2; }
        validate_pr_merge "$2" "$3" "$4"
        ;;
    checkout-rocshmem)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        checkout_rocshmem "$2"
        ;;
    build-wheel)
        [[ "$#" -eq 6 ]] || { usage; exit 2; }
        build_wheel "$2" "$3" "$4" "$5" "$6"
        ;;
    run-pr-tests)
        [[ "$#" -eq 4 ]] || { usage; exit 2; }
        run_pr_tests "$2" "$3" "$4"
        ;;
    run-nightly-tests)
        [[ "$#" -eq 4 ]] || { usage; exit 2; }
        run_nightly_tests "$2" "$3" "$4"
        ;;
    *)
        usage
        exit 2
        ;;
esac
