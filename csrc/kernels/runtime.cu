#include "hip/hip_runtime.h"
#include <cstring>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "shmem_wrapper.cuh"

namespace deep_ep {

namespace intranode {

template <int kNumRanks> 
__global__ void barrier(int **barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int **barrier_signal_ptrs, int rank, int num_ranks, hipStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                                                 \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank);                                \
    break

    SETUP_LAUNCH_CONFIG(1, kWarpSize, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

namespace internode {

#ifndef DISABLE_ROCSHMEM
shmem_team_t        cpu_rdma_team = EP_SHMEM_TEAM_INVALID;
shmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    shmemx_uniqueid_t unique_id;
    shmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(shmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(shmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    shmemx_uniqueid_t  root_unique_id;
    shmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(shmemx_uniqueid_t));
    shmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    shmemx_init_attr(EP_SHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        shmem_barrier_all();
        EP_HOST_ASSERT(cpu_rdma_team == EP_SHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(shmem_team_split_strided(
                               EP_SHMEM_TEAM_WORLD, rank % NUM_MAX_NVL_PEERS,
                               NUM_MAX_NVL_PEERS, num_ranks / NUM_MAX_NVL_PEERS,
                               &cpu_rdma_team_config, 0, &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != EP_SHMEM_TEAM_INVALID);

#ifdef FORCE_DUSHMEM_API
        dushmemi_device_host_state_t* dev_state_ptr = nullptr;
        CUDA_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr), dushmemi_device_state_d));
        bool ibgda_is_initialized = false;
        CUDA_CHECK(hipMemcpy(&dev_state_ptr->ibgda_is_initialized, &ibgda_is_initialized, sizeof(bool), hipMemcpyHostToDevice));
#endif
    }

    shmem_barrier_all();
    return shmem_my_pe();
}

void *alloc(size_t size, size_t alignment) {
    return shmem_align(size, alignment);
}

void free(void *ptr) {
    shmem_free(ptr);
}

void barrier() {
    shmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != EP_SHMEM_TEAM_INVALID) {
        shmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = EP_SHMEM_TEAM_INVALID;
    }
    shmem_finalize();
}
#endif

} // namespace internode

} // namespace deep_ep
