#pragma once
/*
 * Temporary wrapper for for platform specific NVSHMEM and rocSHMEM functions.
 * Once hipify or hipify-torch fully supports this mapping, this file has to be
 * removed and according nvshmem* functions restored.
 */
#ifndef DISABLE_ROCSHMEM

#include "configs.cuh"

#ifndef FORCE_NVSHMEM_API
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#else
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <infiniband/mlx5dv.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#endif

namespace deep_ep::internode {

// rocSHMEM wrapper
#ifndef FORCE_NVSHMEM_API
using shmem_team_t = rocshmem::rocshmem_team_t;
using shmem_team_config_t = rocshmem::rocshmem_team_config_t;
const shmem_team_t EP_SHMEM_TEAM_INVALID = rocshmem::ROCSHMEM_TEAM_INVALID;
inline shmem_team_t& EP_SHMEM_TEAM_WORLD = rocshmem::ROCSHMEM_TEAM_WORLD;

using shmemx_uniqueid_t = rocshmem::rocshmem_uniqueid_t;
using shmemx_init_attr_t = rocshmem::rocshmem_init_attr_t;

constexpr auto EP_SHMEMX_INIT_WITH_UNIQUEID = rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID;

__host__ inline int shmemx_get_uniqueid(shmemx_uniqueid_t *uid) {
    return rocshmem::rocshmem_get_uniqueid(uid);
}

__host__ inline int shmemx_set_attr_uniqueid_args(int rank, int nranks,
    shmemx_uniqueid_t *uid, shmemx_init_attr_t *attr) {
    return rocshmem::rocshmem_set_attr_uniqueid_args(rank, nranks, uid, attr);
}

__host__ inline int shmemx_init_attr(unsigned int flags, shmemx_init_attr_t *attr) {
    return rocshmem::rocshmem_init_attr(flags, attr);
}

__host__ inline int shmem_team_split_strided(shmem_team_t parent_team,
    int start, int stride, int size,
    const shmem_team_config_t *config,
    long config_mask, shmem_team_t *new_team) {
    return rocshmem::rocshmem_team_split_strided(parent_team, start, stride, size, config, config_mask, new_team);
}

__host__ inline void shmem_barrier_all() {
    rocshmem::rocshmem_barrier_all();
}

__device__ inline void shmem_device_barrier_all() {
    rocshmem::rocshmem_barrier_all();
}

__device__ inline void shmem_barrier(shmem_team_t team) {
    rocshmem::rocshmem_ctx_barrier(rocshmem::ROCSHMEM_CTX_DEFAULT, team);
}

__host__ inline int shmem_my_pe(){
    return rocshmem::rocshmem_my_pe();
}

__host__ inline void shmem_free(void *ptr){
    rocshmem::rocshmem_free(ptr);
}

__host__ inline void* shmem_align(const size_t alignment, const size_t size) {
    auto alloc_size = ALIGN(size, alignment);
    return rocshmem::rocshmem_malloc(alloc_size);
}

__host__ inline void shmem_finalize() {
    rocshmem::rocshmem_finalize();
}

__host__ inline void shmem_team_destroy(shmem_team_t team) {
    rocshmem::rocshmem_team_destroy(team);
}

__device__ inline void shmem_fence() {
    rocshmem::rocshmem_fence();
}

__device__ inline void shmem_int_put_nbi(
    int *dest, const int *source, size_t nelems, int pe) {
    rocshmem::rocshmem_int_put_nbi(dest, source, nelems, pe);
}

__device__ inline void shmemx_int_put_nbi_warp(
    int *dest, const int *source, size_t nelems, int pe) {
    rocshmem::rocshmem_int_put_nbi_wave(dest, source, nelems, pe);
}

__device__ inline void shmemx_int8_put_nbi_warp(
    signed char *dest, const signed char *source, size_t nelems, int pe) {
    rocshmem::rocshmem_schar_put_nbi_wave(dest, source, nelems, pe);
}

__device__ inline void shmem_signal_op_add(uint64_t *dest, uint64_t value, int pe) {
    rocshmem::rocshmem_ulong_atomic_add(dest, value, pe);
}

__device__ inline void shmem_long_atomic_add(
    long *dest, long value, int pe) {
    rocshmem::rocshmem_long_atomic_add(dest, value, pe);
}

#if defined(ROCM_USE_MULTIQP)
__device__ inline void shmemx_int8_put_nbi_warp_dp(
    signed char *dest, const signed char *source, size_t nelems, int qp_idx, int pe) {
    rocshmem::rocshmem_schar_put_nbi_wave_dp(dest, source, nelems, qp_idx, pe);
}

__device__ inline void shmem_long_atomic_add_dp(
    long *dest, long value, int qp_idx, int pe) {
    rocshmem::rocshmem_long_atomic_add_dp(dest, value, qp_idx, pe);
}
#endif

#if !defined(ROCM_DISABLE_CTX)
using shmem_ctx_t = rocshmem::rocshmem_ctx_t;

__device__ inline int shmem_wg_ctx_create(shmem_ctx_t *ctx) {
    return rocshmem::rocshmem_wg_ctx_create(0, ctx);
}

__device__ inline void shmem_wg_ctx_destroy(shmem_ctx_t *ctx) {
    rocshmem::rocshmem_wg_ctx_destroy(ctx);
}

__device__ inline void shmem_ctx_quiet(shmem_ctx_t ctx) {
    rocshmem::rocshmem_ctx_quiet(ctx);
}

__device__ inline void shmem_ctx_ulong_atomic_add(
    shmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe) {
    rocshmem::rocshmem_ctx_ulong_atomic_add(ctx, dest, value, pe);
}

__device__ inline void shmem_ctx_long_atomic_add(
    shmem_ctx_t ctx, long *dest, long value, int pe) {
    rocshmem::rocshmem_ctx_long_atomic_add(ctx, dest, value, pe);
}

__device__ inline void shmem_ctx_schar_put_nbi_warp(
    shmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems, int pe) {
    rocshmem::rocshmem_ctx_schar_put_nbi_wave(ctx, dest, source, nelems, pe);
}

__device__ inline void shmem_ctx_int_put_nbi_warp(
    shmem_ctx_t ctx, int *dest, const int *source, size_t nelems, int pe) {
    rocshmem::rocshmem_ctx_int_put_nbi_wave(ctx, dest, source, nelems, pe);
}

#endif

#else

// NVSHMEM wrapper
#ifndef ROCM_DISABLE_CTX
#define ROCM_DISABLE_CTX
#endif

using shmem_team_t = nvshmem_team_t;
using shmem_team_config_t = nvshmem_team_config_t;
using shmemx_uniqueid_t = nvshmemx_uniqueid_t;
using shmemx_init_attr_t = nvshmemx_init_attr_t;
const shmem_team_t EP_SHMEM_TEAM_INVALID = NVSHMEM_TEAM_INVALID;
const shmem_team_t EP_SHMEM_TEAM_WORLD = NVSHMEM_TEAM_WORLD;
constexpr auto EP_SHMEMX_INIT_WITH_UNIQUEID = NVSHMEMX_INIT_WITH_UNIQUEID;


__host__ inline int shmemx_get_uniqueid(shmemx_uniqueid_t *uid) {
    return nvshmemx_get_uniqueid(uid);
}

__host__ inline int shmemx_set_attr_uniqueid_args(int rank, int nranks,
    shmemx_uniqueid_t *uid, shmemx_init_attr_t *attr) {
    return nvshmemx_set_attr_uniqueid_args(rank, nranks, uid, attr);
}


__host__ inline int shmemx_init_attr(unsigned int flags, shmemx_init_attr_t *attr) {
    return nvshmemx_init_attr(flags, attr);
}

__host__ inline int shmem_team_split_strided(shmem_team_t parent_team,
    int start, int stride, int size,
    const shmem_team_config_t *config,
    long config_mask, shmem_team_t *new_team) {
    return nvshmem_team_split_strided(parent_team, start, stride, size, config, config_mask, new_team);
}

__host__ inline void shmem_barrier_all() {
    nvshmem_barrier_all();
}

__device__ inline void shmem_device_barrier_all() {
    nvshmem_barrier_all();
}

__device__ inline void shmem_barrier(shmem_team_t team) {
    void(nvshmem_barrier(team));
}

__host__ inline int shmem_my_pe(){
    return nvshmem_my_pe();
}

__host__ inline void shmem_free(void *ptr){
    nvshmem_free(ptr);
}

__host__ inline void* shmem_align(const size_t alignment, const size_t size) {
    return nvshmem_align(size, alignment);
}

__host__ inline void shmem_finalize() {
    nvshmem_finalize();
}

__host__ inline void shmem_team_destroy(shmem_team_t team) {
    nvshmem_team_destroy(team);
}

__device__ inline void shmem_fence() {
    nvshmem_fence();
}

__device__ inline void shmem_int_put_nbi(
    int *dest, const int *source, size_t nelems, int pe) {
    nvshmem_int_put_nbi(dest, source, nelems, pe);
}

__device__ inline void shmemx_int_put_nbi_warp(
    int *dest, const int *source, size_t nelems, int pe) {
    nvshmemx_int_put_nbi_warp(dest, source, nelems, pe);
}

__device__ inline void shmemx_int8_put_nbi_warp(
    signed char *dest, const signed char *source, size_t nelems, int pe) {
    nvshmemx_int8_put_nbi_warp(dest, source, nelems, pe);
}

__device__ inline void shmem_signal_op_add(
    uint64_t *dest, uint64_t value, int pe) {
    nvshmemx_signal_op(dest, value, NVSHMEM_SIGNAL_ADD, pe);
}

__device__ inline void shmem_ulong_atomic_add(
    uint64_t *dest, uint64_t value, int pe) {
    nvshmem_ulong_atomic_add(dest, value, pe);
}

__device__ inline void shmem_long_atomic_add(
    long *dest, long value, int pe) {
    // nvshmem_##Name##_atomic_add(dest, value, pe);
    nvshmem_long_atomic_add(dest, value, pe);
}

#endif

} // namespace deep_ep::internode

#endif
