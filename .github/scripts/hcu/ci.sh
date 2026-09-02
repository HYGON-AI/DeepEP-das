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

runner_group_id() {
    local runner_temp group_id
    runner_temp="$(realpath -- "${RUNNER_TEMP:?RUNNER_TEMP is required}")"
    group_id="$(stat -c '%g' -- "${runner_temp}")"
    [[ "${group_id}" =~ ^[1-9][0-9]*$ ]] || die "unsafe runner group ID: ${group_id}"
    printf '%s\n' "${group_id}"
}

runner_owner() {
    local runner_temp owner
    runner_temp="$(realpath -- "${RUNNER_TEMP:?RUNNER_TEMP is required}")"
    owner="$(stat -c '%u:%g' -- "${runner_temp}")"
    [[ "${owner}" =~ ^[1-9][0-9]*:[0-9]+$ ]] || die "unsafe runner ownership: ${owner}"
    printf '%s\n' "${owner}"
}

validate_workspace_path() {
    local path workspace
    path="$(realpath -m -- "$1")"
    workspace="$(realpath -m -- "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}")"

    case "${path}" in
        "${workspace}"|"${workspace}"/*) ;;
        *) die "path is outside the runner workspace: ${path}" ;;
    esac
    printf '%s\n' "${path}"
}

self_docker_container() {
    local current_hostname container_id container_hostname
    [[ -f /.dockerenv ]] || return 1
    current_hostname="$(hostname)"

    if docker inspect --type container "${current_hostname}" >/dev/null 2>&1; then
        printf '%s\n' "${current_hostname}"
        return 0
    fi

    while IFS= read -r container_id; do
        [[ "${container_id}" =~ ^[0-9a-f]{12,64}$ ]] || continue
        container_hostname="$(
            docker inspect --format '{{.Config.Hostname}}' "${container_id}" 2>/dev/null || true
        )"
        if [[ "${container_hostname}" == "${current_hostname}" ]]; then
            printf '%s\n' "${container_id}"
            return 0
        fi
    done < <(docker ps --quiet)
    return 1
}

prepare_ci_public_root() {
    local public_root runner_group
    public_root="$(realpath -m -- "$1")"

    case "${public_root}" in
        /ci_public/deepep-das/*) ;;
        *) die "refusing to prepare unexpected CI public root: ${public_root}" ;;
    esac

    runner_group="$(runner_group_id)"
    umask 0002
    mkdir -p -- "${public_root}"
    if [[ "$(stat -c '%g' -- "${public_root}")" != "${runner_group}" ]]; then
        chgrp -- "${runner_group}" "${public_root}"
    fi
    if [[ "$(stat -c '%a' -- "${public_root}")" != "2775" ]]; then
        chmod 2775 -- "${public_root}"
    fi
    [[ -d "${public_root}" && -w "${public_root}" ]] || \
        die "CI public root is not writable: ${public_root}"

    append_summary "- CI public root: ${public_root} (group ${runner_group}, mode 2775)"
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

restore_workspace_with_container() {
    local workspace image owner runner_container
    workspace="$(validate_workspace_path "$1")"
    image="$2"
    owner="$(runner_owner)"

    if runner_container="$(self_docker_container)"; then
        docker exec --user 0 "${runner_container}" \
            /bin/chown -R -- "${owner}" "${workspace}"
    else
        docker run --rm \
            --user 0:0 \
            --volume "${workspace}:/workspace" \
            --entrypoint /bin/chown \
            "${image}" \
            -R -- "${owner}" /workspace
    fi
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

prepare_container_source() {
    local source_dir source_ref repository token auth_header source_url source_sha
    source_dir="$(resolve_dir "$1")"
    source_ref="${DEEPEP_SOURCE_REF:?DEEPEP_SOURCE_REF is required}"
    repository="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
    token="${HYGON_AI_CI_TOKEN:?HYGON_AI_CI_TOKEN is required}"

    [[ "${repository}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || \
        die "invalid GitHub repository: ${repository}"
    [[ "${source_ref}" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$ ]] || \
        die "invalid source ref: ${source_ref}"
    source_url="https://github.com/${repository}.git"
    auth_header="$(printf 'x-access-token:%s' "${token}" | base64 | tr -d '\r\n')"

    rm -rf -- "${source_dir:?}/.git"
    git -C "${source_dir}" init --quiet
    git -C "${source_dir}" remote add origin "${source_url}"
    git -C "${source_dir}" \
        -c "http.https://github.com/.extraheader=Authorization: Basic ${auth_header}" \
        fetch --no-tags --depth=2 origin -- "${source_ref}"
    git -C "${source_dir}" checkout --force --detach FETCH_HEAD
    git -C "${source_dir}" clean -ffd

    source_sha="$(git -C "${source_dir}" rev-parse HEAD)"
    [[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || die "invalid checked-out source SHA: ${source_sha}"
    if [[ "${source_ref}" =~ ^[0-9a-f]{40}$ ]]; then
        [[ "${source_sha}" == "${source_ref}" ]] || die "source checkout does not match the requested commit"
    fi
    unset token auth_header
    append_summary "- source ref: checked out and verified inside the build container"
}

checkout_rocshmem() {
    local source_dir configured_url gitlink_sha recorded_gitlink_sha token auth_header
    source_dir="$(resolve_dir "$1")"
    token="${HYGON_AI_CI_TOKEN:?HYGON_AI_CI_TOKEN is required}"

    configured_url="$(git -C "${source_dir}" config -f .gitmodules --get "submodule.${ROCSHMEM_PATH}.url")"
    [[ "${configured_url}" == "${ROCSHMEM_URL}" ]] || die "unexpected rocSHMEM submodule URL: ${configured_url}"

    if git -C "${source_dir}" cat-file -e 'HEAD^{commit}' >/dev/null 2>&1; then
        recorded_gitlink_sha="$(
            git -C "${source_dir}" ls-tree HEAD "${ROCSHMEM_PATH}" | awk '{print $3}'
        )"
        if [[ -n "${DEEPEP_ROCSHMEM_SHA:-}" ]]; then
            [[ "${recorded_gitlink_sha}" == "${DEEPEP_ROCSHMEM_SHA}" ]] || \
                die "rocSHMEM gitlink does not match the runner checkout"
        fi
        gitlink_sha="${recorded_gitlink_sha}"
    else
        gitlink_sha="${DEEPEP_ROCSHMEM_SHA:?DEEPEP_ROCSHMEM_SHA is required without source Git metadata}"
    fi
    [[ "${gitlink_sha}" =~ ^[0-9a-f]{40}$ ]] || die "invalid rocSHMEM gitlink SHA: ${gitlink_sha}"

    auth_header="$(printf 'x-access-token:%s' "${token}" | base64 | tr -d '\r\n')"
    if git -C "${source_dir}" cat-file -e 'HEAD^{commit}' >/dev/null 2>&1; then
        git -C "${source_dir}" \
            -c "http.https://github.com/.extraheader=Authorization: Basic ${auth_header}" \
            submodule update --init --recursive -- "${ROCSHMEM_PATH}"
    else
        rm -rf -- "${source_dir:?}/${ROCSHMEM_PATH}"
        git -c "http.https://github.com/.extraheader=Authorization: Basic ${auth_header}" \
            clone --no-checkout -- "${ROCSHMEM_URL}" "${source_dir}/${ROCSHMEM_PATH}"
        git -C "${source_dir}/${ROCSHMEM_PATH}" \
            -c "http.https://github.com/.extraheader=Authorization: Basic ${auth_header}" \
            checkout --detach "${gitlink_sha}"
        git -C "${source_dir}/${ROCSHMEM_PATH}" \
            -c "http.https://github.com/.extraheader=Authorization: Basic ${auth_header}" \
            submodule update --init --recursive
    fi

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

    rm -rf -- \
        "${source_dir}/build_" \
        "${source_dir}/dist" \
        "${source_dir}/third-party/rocshmem/build" \
        "${source_dir}/third-party/rocshmem_install" \
        "${output_dir}"
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
    if git -C "${source_dir}" cat-file -e 'HEAD^{commit}' >/dev/null 2>&1; then
        git_commit="$(git -C "${source_dir}" rev-parse --short=6 HEAD)"
        if [[ -n "${DEEPEP_SOURCE_SHA:-}" ]]; then
            [[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${DEEPEP_SOURCE_SHA}" ]] || \
                die "source commit does not match the runner checkout"
        fi
    else
        [[ "${DEEPEP_SOURCE_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || \
            die "DEEPEP_SOURCE_SHA is required without source Git metadata"
        git_commit="${DEEPEP_SOURCE_SHA:0:6}"
    fi

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

container_ci() {
    local controller_dir source_dir build_variant torch_version dtk_package mode output_dir log_dir
    controller_dir="$(resolve_dir "$1")"
    source_dir="$(resolve_dir "$2")"
    build_variant="$3"
    torch_version="$4"
    dtk_package="$5"
    mode="$6"
    output_dir="$(realpath -m -- "$7")"
    log_dir="$(realpath -m -- "$8")"

    case "${mode}" in
        build|pr|nightly) ;;
        *) die "unknown container CI mode: ${mode}" ;;
    esac
    case "${output_dir}" in "${source_dir}"/*) ;; *) die "invalid output directory" ;; esac
    case "${log_dir}" in "${source_dir}"/*) ;; *) die "invalid log directory" ;; esac

    git config --global --add safe.directory "${source_dir}"
    prepare_container_source "${source_dir}"
    mkdir -p -- "${log_dir}"
    git config --global --add safe.directory "${source_dir}/${ROCSHMEM_PATH}"
    git -C "${source_dir}" rev-parse HEAD > "${log_dir}/source-sha.log"
    if [[ -n "${DEEPEP_PR_BASE_SHA:-}" || -n "${DEEPEP_PR_HEAD_SHA:-}" ]]; then
        [[ -n "${DEEPEP_PR_BASE_SHA:-}" && -n "${DEEPEP_PR_HEAD_SHA:-}" ]] || \
            die "both PR base and head SHAs are required"
        validate_pr_merge "${source_dir}" "${DEEPEP_PR_BASE_SHA}" "${DEEPEP_PR_HEAD_SHA}"
    fi
    checkout_rocshmem "${source_dir}" 2>&1 | tee "${log_dir}/submodule.log"
    build_wheel \
        "${source_dir}" "${build_variant}" "${torch_version}" "${dtk_package}" "${output_dir}" \
        2>&1 | tee "${log_dir}/build.log"

    case "${mode}" in
        build) ;;
        pr) run_pr_tests "${source_dir}" "${output_dir}" "${log_dir}" ;;
        nightly) run_nightly_tests "${source_dir}" "${output_dir}" "${log_dir}" ;;
    esac
}

run_ci_container() {
    local controller_dir source_dir image build_variant torch_version dtk_package mode output_dir log_dir
    local workspace container_controller container_source container_output container_log
    local status copy_status nightly_temp container_name host_container_log
    local -a docker_args
    controller_dir="$(validate_workspace_path "$1")"
    source_dir="$(validate_workspace_path "$2")"
    image="$3"
    build_variant="$4"
    torch_version="$5"
    dtk_package="$6"
    mode="$7"
    output_dir="$(validate_workspace_path "$8")"
    log_dir="$(validate_workspace_path "$9")"
    workspace="$(realpath -- "${GITHUB_WORKSPACE}")"
    container_controller="/workspace/${controller_dir#"${workspace}"/}"
    container_source="/workspace/${source_dir#"${workspace}"/}"
    container_output="/workspace/${output_dir#"${workspace}"/}"
    container_log="/workspace/${log_dir#"${workspace}"/}"

    case "${mode}" in
        build|pr|nightly) ;;
        *) die "unknown container CI mode: ${mode}" ;;
    esac
    mkdir -p -- "${log_dir}"
    docker_args=(
        --user 0:0
        --workdir /workspace
        --volume /opt/hyhal:/opt/hyhal:ro
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --shm-size 16g
        --cap-add SYS_PTRACE
        --env PIP_INDEX_URL
        --env PIP_TRUSTED_HOST
        --env CISMI_TIMEOUT
        --env CI
        --env FREE_BUILD_DIR
        --env HYGON_AI_CI_TOKEN
        --env GITHUB_REPOSITORY
        --env GITHUB_RUN_ID
        --env GITHUB_RUN_ATTEMPT
        --env BUILD_VARIANT
        --env TORCH_VERSION
        --env DEEPEP_SOURCE_REF
        --env DEEPEP_PR_BASE_SHA
        --env DEEPEP_PR_HEAD_SHA
        --entrypoint /bin/bash
    )
    if [[ "${mode}" == "build" ]]; then
        docker_args+=(--privileged)
    fi

    nightly_temp=""
    if [[ "${mode}" == "nightly" ]]; then
        [[ -n "${DEEPEP_NIGHTLY_CONFIG_ROOT:-}" ]] || die "DEEPEP_NIGHTLY_CONFIG_ROOT is required"
        [[ -d "${DEEPEP_NIGHTLY_CONFIG_ROOT}" ]] || \
            die "Nightly configuration root does not exist: ${DEEPEP_NIGHTLY_CONFIG_ROOT}"
        nightly_temp="$(mktemp -d "${RUNNER_TEMP:?RUNNER_TEMP is required}/deepep-nightly.XXXXXX")"
        cp -R -- "${DEEPEP_NIGHTLY_CONFIG_ROOT}/." "${nightly_temp}/"
        docker_args+=(
            --env DEEPEP_TEST_PROFILE
            --env DEEPEP_NIGHTLY_CONFIG_ROOT=/tmp/ci-nightly
        )
    fi

    container_name="deepep-${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}-${GITHUB_RUN_ATTEMPT:-1}"
    container_name+="-${build_variant}-torch-${torch_version//./-}"
    docker rm --force "${container_name}" >/dev/null 2>&1 || true
    docker create --name "${container_name}" "${docker_args[@]}" "${image}" \
        "${container_controller}/.github/scripts/hcu/ci.sh" container-ci \
        "${container_controller}" "${container_source}" \
        "${build_variant}" "${torch_version}" "${dtk_package}" "${mode}" \
        "${container_output}" "${container_log}" \
        >/dev/null
    docker cp "${workspace}/." "${container_name}:/workspace/"
    if [[ -n "${nightly_temp}" ]]; then
        docker cp "${nightly_temp}" "${container_name}:/tmp/ci-nightly"
    fi

    host_container_log="$(mktemp "${RUNNER_TEMP}/deepep-container.XXXXXX.log")"
    set +e
    docker start --attach "${container_name}" 2>&1 | tee "${host_container_log}"
    status="${PIPESTATUS[0]}"
    set -e

    copy_status=0
    mkdir -p -- "${log_dir}"
    docker cp "${container_name}:${container_log}/." "${log_dir}/" || copy_status=1
    cp -- "${host_container_log}" "${log_dir}/container.log"
    mkdir -p -- "${output_dir}"
    if docker cp "${container_name}:${container_output}/." "${output_dir}/" 2>/dev/null; then
        :
    elif [[ "${status}" -eq 0 ]]; then
        copy_status=1
    fi
    docker rm --force "${container_name}" >/dev/null
    rm -f -- "${host_container_log}"
    if [[ -n "${nightly_temp}" ]]; then
        rm -rf -- "${nightly_temp}"
    fi
    if [[ "${status}" -eq 0 && "${copy_status}" -ne 0 ]]; then
        die "CI container succeeded but its output could not be copied to the runner workspace"
    fi
    return "${status}"
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

save_ci_output() {
    local source_dir public_root destination parent_dir destination_name staging git_sha runner_group
    local -a wheels
    source_dir="$(realpath -m -- "$1")"
    public_root="$(realpath -- "${CI_PUBLIC_ROOT:-/ci_public}")"
    destination="$(realpath -m -- "$2")"

    [[ -d "${public_root}" && -w "${public_root}" ]] || \
        die "CI public root is not a writable directory: ${public_root}"
    case "${destination}" in
        "${public_root}"/*) ;;
        *) die "CI output directory must be inside ${public_root}: ${destination}" ;;
    esac
    [[ ! -e "${destination}" ]] || die "refusing to overwrite CI output: ${destination}"

    parent_dir="$(dirname -- "${destination}")"
    destination_name="$(basename -- "${destination}")"
    umask 0002
    mkdir -p -- "${parent_dir}"
    staging="$(mktemp -d "${parent_dir}/.${destination_name}.tmp.XXXXXX")"

    if [[ -d "${source_dir}/final" ]]; then
        shopt -s nullglob
        wheels=("${source_dir}"/final/*.whl)
        shopt -u nullglob
        if (( ${#wheels[@]} > 0 )); then
            mkdir -p -- "${staging}/wheels"
            cp -- "${wheels[@]}" "${staging}/wheels/"
        fi
    fi
    if [[ -d "${source_dir}/ci-logs" ]]; then
        cp -R -- "${source_dir}/ci-logs" "${staging}/"
    fi

    git_sha="unknown"
    if git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git_sha="$(git -C "${source_dir}" rev-parse HEAD)"
    elif [[ -f "${source_dir}/ci-logs/source-sha.log" ]]; then
        git_sha="$(tr -d '\r\n' < "${source_dir}/ci-logs/source-sha.log")"
        [[ "${git_sha}" =~ ^[0-9a-f]{40}$ ]] || die "invalid saved source SHA: ${git_sha}"
    fi
    {
        printf 'repository=%s\n' "${GITHUB_REPOSITORY:-unknown}"
        printf 'source_sha=%s\n' "${git_sha}"
        printf 'run_id=%s\n' "${GITHUB_RUN_ID:-unknown}"
        printf 'run_attempt=%s\n' "${GITHUB_RUN_ATTEMPT:-unknown}"
        printf 'build_variant=%s\n' "${BUILD_VARIANT:-unknown}"
        printf 'torch_version=%s\n' "${TORCH_VERSION:-unknown}"
        printf 'test_profile=%s\n' "${DEEPEP_TEST_PROFILE:-not-applicable}"
    } > "${staging}/metadata.txt"
    (
        cd "${staging}"
        while IFS= read -r -d '' file; do
            sha256sum -- "${file}"
        done < <(find . -type f ! -name SHA256SUMS -print0 | sort -z)
    ) > "${staging}/SHA256SUMS"

    runner_group="$(runner_group_id)"
    chgrp -R -- "${runner_group}" "${staging}"
    find "${staging}" -type d -exec chmod 2775 {} +
    find "${staging}" -type f -exec chmod 0664 {} +
    mv -- "${staging}" "${destination}"
    append_summary "- CI output directory: ${destination}"
    printf '%s\n' "${destination}"
}

usage() {
    cat <<'EOF'
Usage:
  ci.sh prepare-ci-public-root <public-root>
  ci.sh restore-workspace <workspace>
  ci.sh restore-workspace-with-container <workspace> <image>
  ci.sh validate-pr-merge <source-dir> <base-sha> <head-sha>
  ci.sh checkout-rocshmem <source-dir>
  ci.sh build-wheel <source-dir> <standard|shca> <torch-version> <dtk-package> <output-dir>
  ci.sh container-ci <controller-dir> <source-dir> <standard|shca> <torch-version> <dtk-package> <build|pr|nightly> <output-dir> <log-dir>
  ci.sh run-ci-container <controller-dir> <source-dir> <image> <standard|shca> <torch-version> <dtk-package> <build|pr|nightly> <output-dir> <log-dir>
  ci.sh run-pr-tests <source-dir> <wheel-dir> <log-dir>
  ci.sh run-nightly-tests <source-dir> <wheel-dir> <log-dir>
  ci.sh save-ci-output <source-dir> <destination-dir>
EOF
}

command="${1:-}"
case "${command}" in
    prepare-ci-public-root)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        prepare_ci_public_root "$2"
        ;;
    restore-workspace)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        restore_workspace "$2"
        ;;
    restore-workspace-with-container)
        [[ "$#" -eq 3 ]] || { usage; exit 2; }
        restore_workspace_with_container "$2" "$3"
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
    container-ci)
        [[ "$#" -eq 9 ]] || { usage; exit 2; }
        container_ci "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"
        ;;
    run-ci-container)
        [[ "$#" -eq 10 ]] || { usage; exit 2; }
        run_ci_container "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}"
        ;;
    run-pr-tests)
        [[ "$#" -eq 4 ]] || { usage; exit 2; }
        run_pr_tests "$2" "$3" "$4"
        ;;
    run-nightly-tests)
        [[ "$#" -eq 4 ]] || { usage; exit 2; }
        run_nightly_tests "$2" "$3" "$4"
        ;;
    save-ci-output)
        [[ "$#" -eq 3 ]] || { usage; exit 2; }
        save_ci_output "$2" "$3"
        ;;
    *)
        usage
        exit 2
        ;;
esac
