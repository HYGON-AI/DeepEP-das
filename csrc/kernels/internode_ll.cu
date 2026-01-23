#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "buffer.cuh"
#include "utils.cuh"
// #include <cooperative_groups.h>
#include <iostream>

#include "hip/hip_runtime.h"

#include "shmem_wrapper.cuh"

namespace deep_ep {

namespace internode_ll {

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

__device__ void grid_barrier(int* global_counter, int num_blocks) {
    volatile int ret;
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0 ) {
        ret = __hip_atomic_fetch_add(&global_counter[0], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(global_counter, __ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT) != num_blocks);
    }
    __syncthreads();
}
template <typename dtype_t>
__host__ __device__ dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}


template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int64_t* clean_0, int num_clean_int_0,
                                         int64_t* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    if (threadIdx.x == 0)
        internode::shmem_device_barrier_all();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure low-latency mode work
    if (threadIdx.x == 0)
        internode::shmem_device_barrier_all();
}

void clean_low_latency_buffer(int64_t* clean_0, int num_clean_int_0,
                              int64_t* clean_1, int num_clean_int_1,
                              hipStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}

__device__ __forceinline__ void 
internode_ll_putmem_nbi(void* dst_ptr, void* src_ptr,
                        int num_ranks, int dst_rank, int expert_idx,
                        int msg_bytes) {
#if defined(FORCE_NVSHMEM_API)
        internode::shmemx_int8_put_nbi_warp(
            reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(src_ptr),
            msg_bytes, dst_rank);
#else
    #if defined(ROCM_DISABLE_MULTIQP)
        internode::shmemx_int8_put_nbi_warp(
            reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(src_ptr),
            msg_bytes, dst_rank);
    #else
        internode::shmemx_int8_put_nbi_warp_dp(
            reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(src_ptr),
            msg_bytes, (expert_idx + 1) * num_ranks + dst_rank, dst_rank);
    #endif
#endif // defined(FORCE_NVSHMEM_API)
}

__device__ __forceinline__ void 
internode_ll_long_atomic_add(long* dest, const long &value, 
                             int num_ranks, int dst_rank, int expert_idx) {
#if defined(FORCE_DUSHMEM_API)
        internode::shmem_long_atomic_add(dest, value, dst_rank);
#else
        #if defined(ROCM_DISABLE_MULTIQP)
        internode::shmem_long_atomic_add(dest, value, dst_rank);
        #else
        internode::shmem_long_atomic_add_dp(dest, value,
            (expert_idx + 1) * num_ranks + dst_rank, dst_rank);
        #endif
#endif // defined(FORCE_DUSHMEM_API)
}

template <bool kUseFP8, bool kUseUE8M0, bool kUseInt8, int kHidden>
__global__ __launch_bounds__(16 * kWarpSize, 1) void
dispatch(void* packed_recv_x, void* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         int* global_atomic_counter,
         void* rdma_recv_x, int64_t* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int64_t* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int num_warp_groups, int num_warps_per_group,
         bool round_scale, int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / kWarpSize, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = FP8_QUANTIZATION_NUM_PER_CHANNEL;
    constexpr int kNumScales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__hip_fp8_storage_t) : sizeof(hip_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    constexpr size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + kNumScales * sizeof(float)) : (kHidden * sizeof(hip_bfloat16)));
    EP_STATIC_ASSERT(num_bytes_per_msg % sizeof(int4) == 0, "Invalid message size");
    constexpr size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 1024 / kWarpSize;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(hip_bfloat16);
        // EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * kWarpSize;
        constexpr int hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index write
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            __shared__ float int8_amaxf[kNumScales];
            if constexpr(kUseInt8) {
                if (thread_id < kNumScales) {
                    int8_amaxf[thread_id] = kFP8Margin;
                }
                __syncthreads();
            }

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr(kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<hip_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }
                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize / kNumPerChannels == 4, "Invalid vectorization");
                    amax = warp_reduce_max<16>(amax);
                    const int scale_offset = i * kNumElemsPerRead / FP8_QUANTIZATION_NUM_PER_CHANNEL;

                    if constexpr(kUseInt8) {
                        // 记录每128个数的最大值
                        int8_amaxf[scale_offset] = fmaxf(amax, int8_amaxf[scale_offset]);
                    } else {
                        calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                        if (lane_id % 16 == 0)
                            rdma_x_scales[scale_offset] = scale_inv;

                        // Cast into send buffer
                        vec_t int2_value;
                        auto fp8x2_values = reinterpret_cast<__hip_fp8x2_storage_t*>(&int2_value);
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerRead; j += 2) {
                            float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                            fp8x2_values[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
                        }
                        rdma_x_vec[i] = int2_value;
                    }
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            __syncthreads();

            if constexpr(kUseInt8) {
                float amax_per_token = kFP8Margin;
                // 并行规约，计算每个token的amax
                for (int s = 0; s < kNumScales; s+=kWarpSize) {
                    int src_idx = s + lane_id;
                    float tmp_amaxf = 0;
                    if(src_idx < kNumScales) {
                        tmp_amaxf = int8_amaxf[src_idx];
                    }
                    tmp_amaxf = warp_reduce_max<kWarpSize>(tmp_amaxf);
                    int8_amaxf[0] = fmaxf(tmp_amaxf, int8_amaxf[0]);
                    __syncthreads();
                }
                amax_per_token = int8_amaxf[0];

                // 根据最大值计算scale
                float scale, scale_inv;
                calculate_int8_scales(amax_per_token, scale, scale_inv);
                if (thread_id == 0) {
                    rdma_x_scales[0] = scale_inv;
                }

                for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                    // Read
                    auto int4_value = __ldg(x_int4 + i);
                    auto bf16_values = reinterpret_cast<hip_bfloat16*>(&int4_value);

                    // Cast into send buffer
                    vec_t int2_value;
                    auto int8_values = reinterpret_cast<int8_t*>(&int2_value);
#pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        auto fp32_value = static_cast<float>(bf16_values[j]);
                        auto fp32_value_scaled = fp32_value * scale;
                        int8_values[j] = static_cast<int8_t>(nearbyintf(fp32_value_scaled));
                    }
                    rdma_x_vec[i] = int2_value;
                }
                __syncthreads();
            }

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = shfl_sync(slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;

                // 通过 shmem_get_p2p_ptr 获取 当前远程指针能否可达
                uint64_t p2p_ptr = internode::shmem_get_p2p_ptr((void*)dst_ptr, rank, dst_rank);
                if (p2p_ptr == 0) {  // RDMA
                    internode_ll_putmem_nbi((void*)dst_ptr, (void*)src_ptr,
                        num_ranks, dst_rank, dst_expert_local_idx,
                        num_bytes_per_msg);
                } else { //  本地 GPU 和 同一计算节点的 其他 GPU 地址
                    // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(p2p_ptr);
                    UNROLLED_WARP_COPY_LL(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                // Increase counter after finishing
                syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    }
    if (warp_id == num_warps - 1) {
        // EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += kWarpSize)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += kWarpSize)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }
        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += kWarpSize) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);

        auto dst_ptr = rdma_recv_count + dst_expert_local_idx * num_ranks + rank;
        // 通过 shmem_get_p2p_ptr 获取 当前远程指针能否可达
        uint64_t p2p_ptr = internode::shmem_get_p2p_ptr((void*)dst_ptr, rank, dst_rank);
        if (p2p_ptr == 0) {  // RDMA
            internode_ll_long_atomic_add(dst_ptr, -num_tokens_sent - 1, 
                                         num_ranks, dst_rank, dst_expert_local_idx);
        } else { //  本地 GPU 和 同一计算节点的 其他 GPU 地址
            st_na_release(reinterpret_cast<int *>(p2p_ptr), -num_tokens_sent - 1);
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    syncwarp();

    // Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE){
        grid_barrier(global_atomic_counter, num_sms);
    }

    // 16 is the max possible number of warps in AMD GPUs
    constexpr int kMaxNumWarps = 1024 / kWarpSize;
    constexpr int num_sync_large_iteration = kMaxNumWarps ;
    __shared__ volatile int sync_large_warp_counters[num_sync_large_iteration];

#pragma unroll
    for (int i = thread_id; i < num_sync_large_iteration; i += blockDim.x) {
        sync_large_warp_counters[i] = 0;
    }
    __syncthreads();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
                                       local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                       src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                                 local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = ALIGN<int>(kNumScales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
                                   local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
                                       (kUseInt8 ? 1 : num_aligned_scales);

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        // EP_DEVICE_ASSERT(num_warps_per_group > 1);

        if (sub_warp_id == 1 and lane_id == 0) {
            while ((num_recv_tokens = ld_acquire_global(reinterpret_cast<int*>(rdma_recv_count + local_expert_idx * num_ranks + src_rank))) == 0);
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
        }

        // no needs to reset because there is no iteration
        if (lane_id == 0){
            volatile int ret = __hip_atomic_fetch_add(&sync_large_warp_counters[warp_group_id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        syncwarp();

        while (sync_large_warp_counters[warp_group_id] < num_warps_per_group);
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_STATIC_ASSERT(kNumScales <= 64, "Invalid hidden size");
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY_LL(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr(kUseFP8) {
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;

                if constexpr(kUseInt8) {
                    if (lane_id == 0) {
                        recv_x_scales[token_idx] = ld_nc_global(src_scales);
                    }
                } else {
                    if (lane_id < kNumScales) {
                        const auto pack_idx = lane_id / num_elems_per_pack;
                        const auto elem_idx = lane_id % num_elems_per_pack;
                        auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                        recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                    }
                    if (lane_id + kWarpSize < kNumScales) {
                        const auto pack_idx = (lane_id + kWarpSize) / num_elems_per_pack;
                        const auto elem_idx = (lane_id + kWarpSize) % num_elems_per_pack;
                        auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + kWarpSize));
                        recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                    }
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* global_atomic_counter,
              void* rdma_recv_x, int64_t* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int64_t* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0, bool use_int8,
              void* workspace, int num_device_sms,
              hipStream_t stream, int phases) {
    constexpr int kNumMaxTopK = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 16 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = dispatch<false, false, false, hidden>; \
if (use_fp8 and not use_ue8m0)             \
    dispatch_func = dispatch<true, false, false, hidden>;   \
if (use_fp8 and use_ue8m0)             \
    dispatch_func = dispatch<true, true, false, hidden>;    \
if (use_int8)             \
    dispatch_func = dispatch<true, false, true, hidden>;    \
LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, \
              global_atomic_counter, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, round_scale, phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(16 * kWarpSize, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int64_t* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* global_atomic_counter,
        int64_t* combine_wait_recv_cost_stats,
        int64_t* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / kWarpSize, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(hip_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    EP_STATIC_ASSERT(kHidden % FP8_QUANTIZATION_NUM_PER_CHANNEL == 0, "Invalid hidden");
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(hip_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // 16 is the max possible number of warps in AMD GPUs
    constexpr int kMaxNumWarps = 1024 / kWarpSize;
    __shared__ volatile int sync_large_warp_counters[kMaxNumWarps];
    if (threadIdx.x==0){
        #pragma unroll
        for (int i = 0; i < kMaxNumWarps; ++i) {
            sync_large_warp_counters[i] = 0;
        }
    }
    __syncthreads();

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += kWarpSize)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                             local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
                                     local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            const auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            
            uint64_t p2p_ptr = internode::shmem_get_p2p_ptr((void*)dst_ptr, rank, dst_rank);
            if (p2p_ptr == 0) {  // RDMA
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy)
                    UNROLLED_WARP_COPY_LL(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);

                internode_ll_putmem_nbi((void*)dst_ptr, (void*)buf_ptr,
                    num_ranks, dst_rank, local_expert_idx,
                    hidden * sizeof(hip_bfloat16));
            } else { //  本地 GPU 和 同一计算节点的 其他 GPU 地址
                // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                const auto* src_int4_ptr = reinterpret_cast<const int4*>(x_int4);
                const auto* dst_int4_ptr = reinterpret_cast<int4*>(p2p_ptr);
                UNROLLED_WARP_COPY_LL(7, lane_id, hidden_bf16_int4, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
            }
        }

        // Put finishing flag
        // EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (lane_id == 0){
            volatile int ret = __hip_atomic_fetch_add(&sync_large_warp_counters[warp_group_id], 1,__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        syncwarp();
        while (sync_large_warp_counters[warp_group_id] < num_warps_per_group);

        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);

            auto dst_ptr = rdma_recv_flag + global_expert_idx;
            // 通过 shmem_get_p2p_ptr 获取 当前远程指针能否可达
            uint64_t p2p_ptr = internode::shmem_get_p2p_ptr((void*)dst_ptr, rank, dst_rank);
            if (p2p_ptr == 0) {  // RDMA
                internode_ll_long_atomic_add(dst_ptr, 1, num_ranks, dst_rank, local_expert_idx);
            } else { //  本地 GPU 和 同一计算节点的 其他 GPU 地址
                st_na_release(reinterpret_cast<int *>(p2p_ptr), 1);
            }

            atomic_add_release_global(atomic_clean_flag, -1);
        }
        syncwarp();
    }

    // Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive and notify PCIe usage
    if (responsible_expert_idx < num_experts) {
        // EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            const auto src_rank = responsible_expert_idx / num_local_experts;
            auto start_time = wall_clock64();
            uint64_t wait_recv_cost = 0;
            while (ld_acquire_global(reinterpret_cast<int*>(rdma_recv_flag + responsible_expert_idx)) == 0  // recv not ready
                   && (wait_recv_cost = wall_clock64() - start_time) <= NUM_TIMEOUT_CYCLES   // not timeout
            );

            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank, responsible_expert_idx % num_local_experts, src_rank);
            }

            if (combine_wait_recv_cost_stats != nullptr) {
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    grid_barrier(global_atomic_counter, num_sms);

    // Reduce tokens with FP8 cast
    // EP_DEVICE_ASSERT(num_topk <= kWarpSize and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (kWarpSize * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_recv_x) +
                    (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type);

                // Reduce
                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<hip_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
            }

            // Write results
            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<hip_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<hip_bfloat16>(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x, int64_t* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* global_atomic_counter,
             int64_t* combine_wait_recv_cost_stats,
             int64_t* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             void* workspace, int num_device_sms, hipStream_t stream,
             int phases, bool zero_copy) {
    constexpr int kNumMaxTopk = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 16 / num_warp_groups; // num_warps_per_group>1, "Requires more than one warp per group"
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine<hidden, kNumMaxTopk>; \
LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              global_atomic_counter, \
              combine_wait_recv_cost_stats, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, phases, zero_copy); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace deep_ep
