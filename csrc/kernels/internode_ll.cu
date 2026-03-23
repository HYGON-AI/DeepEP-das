#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "buffer.cuh"
#include "utils.cuh"
// #include <cooperative_groups.h>
#include <iostream>

#include "hip/hip_runtime.h"

#include "shmem_wrapper.cuh"
#include "internode_ll_logfmt.cuh"

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
#if defined(FORCE_DUSHMEM_API)
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
#endif // defined(FORCE_DUSHMEM_API)
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

/**
 * @brief 将 K 个浮点数（BF16/FP32）量化并打包成 INT2（64位）存储
 * 
 * @tparam kQuantType 量化类型 (1: Int8, 2/3: FP8_E4M3/UE8M0, 4: FP8_E5M2)
 * @tparam kNumElemsPerRead 每次读取的元素数量 (通常为 2, 4, 8)
 * @tparam SrcT 源数据类型 (float 或 __hip_bfloat16)
 * @tparam DstT 目标数据类型 (int2 或 int4)
 * @param src_values 源数据数组 (长度 >= kNumElemsPerRead)
 * @param scale 缩放因子 (将 FP32 值映射到量化范围)
 * @param[out] dst_vec 输出的 64 位向量 (int2 或 int4)
 */
template <int kQuantType, int kNumElemsPerRead, typename SrcT, typename DstT>
__forceinline__ __device__ void pack_quantized_values(
    const SrcT* src_values, float scale, DstT& dst_vec) {

    if constexpr (kQuantType == 1) {
        // INT8 量化
        auto int8_ptr = reinterpret_cast<int8_t*>(&dst_vec);
        #pragma unroll
        for (int j = 0; j < kNumElemsPerRead; ++j) {
            // 如果源是 bfloat16，先提升为 float
            float fp32_value_scaled = static_cast<float>(src_values[j]) * scale;
            // 使用 nearbyintf 进行四舍五入
            int8_ptr[j] = static_cast<int8_t>(nearbyintf(fp32_value_scaled));
        }
    } else {
        // FP8 量化 (E4M3, UE8M0, E5M2)
        // 假设 dst_vec 能容纳 kNumElemsPerRead/2 个 fp8x2 元素
        auto fp8x2_ptr = reinterpret_cast<__hip_fp8x2_storage_t*>(&dst_vec);
        #pragma unroll
        for (int j = 0; j < kNumElemsPerRead; j += 2) {
            // 处理两个元素
            float2 fp32x2 = {static_cast<float>(src_values[j]) * scale, static_cast<float>(src_values[j + 1]) * scale};

            if constexpr (kQuantType == 4) {
                // FP8 E5M2
                fp8x2_ptr[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E5M2);
            } else {
                // FP8 E4M3 或 UE8M0
                fp8x2_ptr[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3);
            }
        }
    }
}

template <int kHidden, int kQuantType=0, int kQuantGroupSize=0, int kMaxNumWarps=16>
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
             bool fp8_round_scale, int phases) {
    // 定义量化类型的枚举
    enum class QuantType {
        None        = 0,        // 不进行量化
        Int8        = 1,        // 采用 Int8 量化
        FP8_E4M3    = 2,        // 采用 FP8 量化 __HIP_E4M3
        FP8_UE8M0   = 3,        // 采用 FP8 量化 DeepseekV3.1的 UE8M0
        FP8_E5M2    = 4         // 采用 FP8 量化 __HIP_E5M2
    };

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
    constexpr bool kUseQuant8Bit = kQuantType > 0;
    constexpr bool kUseUE8M0 = kQuantType == 3; // QuantType::FP8_UE8M0
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = QUANTIZATION_GROUPSIZE;
    constexpr int kNumScales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseQuant8Bit ? sizeof(__hip_fp8_storage_t) : sizeof(hip_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseQuant8Bit, int2, int4>::type;
    constexpr size_t num_bytes_per_msg = sizeof(int4) + 
        (kUseQuant8Bit ? (kHidden + (kQuantGroupSize == 0 ? 4 : kNumScales) * sizeof(float)) : (kHidden * sizeof(hip_bfloat16)));
    EP_STATIC_ASSERT(num_bytes_per_msg % sizeof(int4) == 0, "Invalid message size");
    constexpr size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);

    // Expert counts
    __shared__ int shared_num_tokens_sent_per_expert[kMaxNumWarps];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(hip_bfloat16);
        constexpr int kNumThreadPerGroup = QUANTIZATION_GROUPSIZE / kNumElemsPerRead;
        // EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = num_warps * kWarpSize;
        constexpr int hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index write
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // 用于记录per-channel量化的amax
            __shared__ float channel_amaxf[kNumScales];
            if constexpr(kUseQuant8Bit && kQuantGroupSize == 0) {
                if (thread_id < kNumScales) {
                    channel_amaxf[thread_id] = 0.0;
                }
                __syncthreads();
            }

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr(kUseQuant8Bit) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<hip_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = 0.0, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }
                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize / kNumPerChannels == 4, "Invalid vectorization");
                    amax = warp_reduce_max<kNumThreadPerGroup>(amax);
                    const int scale_offset = i * kNumElemsPerRead / QUANTIZATION_GROUPSIZE;

                    if constexpr(kQuantGroupSize == 0) {
                        // 记录每128个数的最大值
                        channel_amaxf[scale_offset] = fmaxf(amax, channel_amaxf[scale_offset]);
                    } else {
                        calculate_quant8bit_scales<kQuantType>(amax, scale, scale_inv, fp8_round_scale);
                        if (lane_id % kNumThreadPerGroup == 0)
                            rdma_x_scales[scale_offset] = scale_inv;

                        // Cast into send buffer
                        vec_t int2_value;
                        pack_quantized_values<kQuantType, kNumElemsPerRead>(fp32_values, scale, int2_value);
                        rdma_x_vec[i] = int2_value;
                    }
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            __syncthreads();

            if constexpr(kUseQuant8Bit && kQuantGroupSize == 0) {
                float amax_per_token = 0.0;
                // 并行规约，计算每个token的amax
                for (int s = 0; s < kNumScales; s+=kWarpSize) {
                    int src_idx = s + lane_id;
                    float tmp_amaxf = 0;
                    if(src_idx < kNumScales) {
                        tmp_amaxf = channel_amaxf[src_idx];
                    }
                    tmp_amaxf = warp_reduce_max<kWarpSize>(tmp_amaxf);
                    channel_amaxf[0] = fmaxf(tmp_amaxf, channel_amaxf[0]);
                    __syncthreads();
                }
                amax_per_token = channel_amaxf[0];

                // 根据最大值计算scale
                float scale, scale_inv;
                calculate_quant8bit_scales<kQuantType>(amax_per_token, scale, scale_inv, fp8_round_scale);
                if (thread_id == 0) {
                    rdma_x_scales[0] = scale_inv;
                }

                for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                    // Read
                    auto int4_value = __ldg(x_int4 + i);
                    auto bf16_values = reinterpret_cast<hip_bfloat16*>(&int4_value);

                    // Cast into send buffer
                    vec_t int2_value;
                    pack_quantized_values<kQuantType, kNumElemsPerRead>(bf16_values, scale, int2_value);
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
        int expert_count[kMaxNumWarps] = {0};
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
                                       (kQuantGroupSize == 0 ? 1 : num_aligned_scales);

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kMaxNumWarps], shared_recv_token_begin_idx[kMaxNumWarps];

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
            if constexpr(kUseQuant8Bit) {
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;

                if constexpr(kQuantGroupSize == 0) {
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
              int quant_type, int quant_group_size, bool fp8_round_scale,
              void* workspace, int num_device_sms,
              hipStream_t stream, int phases) {
    constexpr int kMaxNumWarps = 16;
    constexpr int kNumMaxTopK = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = kMaxNumWarps / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // 限制groupsize的大小
    EP_HOST_ASSERT(quant_group_size == 0 || quant_group_size == 128);

    /*量化类型枚举
    0 -> None          不量化，保持原始精度
    1 -> Int8          使用 INT8 对称量化
    2 -> FP8_E4M3      使用 FP8 E4M3 格式 (__HIP_E4M3)
    3 -> FP8_UE8M0     使用 DeepSeekV3.1 提出的 UE8M0 格式 (仅支持round_scale=True)
    4 -> FP8_E5M2      使用 FP8 E5M2 格式 (__HIP_E5M2)
    */

#define DISPATCH_LAUNCH_CASE(hidden)                                                \
  {                                                                                 \
    auto dispatch_func = dispatch<hidden, 0, 0, kMaxNumWarps>;                      \
    if (quant_group_size == 0) {                                                    \
        switch (quant_type) {                                                       \
            case 1: dispatch_func = dispatch<hidden, 1, 0, kMaxNumWarps>; break;    \
            case 2: dispatch_func = dispatch<hidden, 2, 0, kMaxNumWarps>; break;    \
            case 3: dispatch_func = dispatch<hidden, 3, 0, kMaxNumWarps>; break;    \
            case 4: dispatch_func = dispatch<hidden, 4, 0, kMaxNumWarps>; break;    \
        }                                                                           \
    } else {                                                                        \
        switch (quant_type) {                                                       \
            case 1: dispatch_func = dispatch<hidden, 1, 128, kMaxNumWarps>; break;  \
            case 2: dispatch_func = dispatch<hidden, 2, 128, kMaxNumWarps>; break;  \
            case 3: dispatch_func = dispatch<hidden, 3, 128, kMaxNumWarps>; break;  \
            case 4: dispatch_func = dispatch<hidden, 4, 128, kMaxNumWarps>; break;  \
        }                                                                           \
    }                                                                               \
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, dispatch_func,                              \
        packed_recv_x, packed_recv_x_scales,                                        \
        packed_recv_src_info, packed_recv_layout_range, packed_recv_count,          \
        global_atomic_counter,                                                      \
        rdma_recv_x, rdma_recv_count, rdma_x, x, topk_idx,                          \
        atomic_counter_per_expert, atomic_finish_counter_per_expert,                \
        next_clean, num_next_clean_int,                                             \
        num_tokens, num_max_dispatch_tokens_per_rank,                               \
        num_topk, num_experts, rank, num_ranks,                                     \
        num_warp_groups, num_warps_per_group, fp8_round_scale, phases);             \
  }                                                                                 \
  break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kMaxNumWarps=16>
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
    const auto num_warps = num_threads / kWarpSize;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(hip_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    EP_STATIC_ASSERT(kHidden % QUANTIZATION_GROUPSIZE == 0, "Invalid hidden");

    /////////////// LogFMT使用 ///////////////
    constexpr int bSupportLogFMT = kUseLogFMT && hidden_bf16_int4 % (kWarpSize * 2) == 0;
    constexpr int kNumSendUnrolls = bSupportLogFMT ? 2 : 1;
    constexpr int kNumRecvUnrolls = bSupportLogFMT ? 2 : 1;
    constexpr int kNumMsgInt4ElemPerWarp = kWarpSize * kNumSendUnrolls; // 每个warp发送的int4元素数据量，即每个warp发送 kNumMsgInt4ElemPerWarp*sizeof(int4)/sizeof(bfloat16)
    EP_STATIC_ASSERT(hidden_bf16_int4 % (kNumSendUnrolls * kWarpSize) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    constexpr int kNumDivisions = kHidden / QUANTIZATION_GROUPSIZE;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(__hip_bfloat162);  // 用于记录数据的最大最小值
    constexpr int kNumSendLogFMTBytes = kNumMsgInt4ElemPerWarp * sizeof(int4);
    constexpr int kNumStages = 3;  // 使用kNumStages>1，则需要的LDS大于64KB
    constexpr int kLogFMTShmemSize = kMaxNumWarps * (kNumStages * kNumSendLogFMTBytes + kNumMetaBytes);
    __shared__ uint8_t smem_buffer[kLogFMTShmemSize];
    /////////////////////////////////////////////

    constexpr size_t num_bytes_per_slot = kHidden * sizeof(hip_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // 初始化用于细粒度warp间同步的计数器数组
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
        // 用于logfmt的LDS
        auto smem_ptr = smem_buffer + warp_id * (kNumStages * kNumSendLogFMTBytes + kNumMetaBytes);
        // 存储logfmt的起始地址，并根据stage_idx进行索引块
        auto logfmt_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * kNumSendLogFMTBytes); });
        // 存储logfmt的最大最小值
        auto meta_buffers = bSupportLogFMT ? reinterpret_cast<__hip_bfloat162*>(smem_ptr + kNumStages * kNumSendLogFMTBytes) : nullptr;
        // 用于多buffer时临时存储
        auto get_num_logfmt_bytes = [&](const int& offset_int4) {
            return min(kNumSendLogFMTBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };
        // 简化从global到LDS的存储写法
        auto logfmt_load_global2lds = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            UNROLLED_WARP_COPY_LL(1, lane_id, num_bytes / sizeof(int4),
                reinterpret_cast<int4 *>(logfmt_buffers[stage_idx]),
                reinterpret_cast<const int4 *>(gmem_ptr),
                ld_direct_global, st_na_global);
        };

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

            // 采用logfmt或者直接拷贝
            uint64_t dst_p2p_ptr = internode::shmem_get_p2p_ptr((void*)dst_ptr, rank, dst_rank);
            int num_send_bytes = hidden * sizeof(hip_bfloat16);

            if (not zero_copy or dst_p2p_ptr != 0) {
                const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                const auto cpy_dst_int4_ptr = dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);

                constexpr int kNumIters = hidden_bf16_int4 / kNumMsgInt4ElemPerWarp;
                EP_STATIC_ASSERT(kNumIters >= 1, "hidden length too small");

                if constexpr (bSupportLogFMT) {
                    // ===== LogFMT 路径：使用 LDS + encode + 多级流水 =====
                    int logfmt_offset_bytes = kNumMetaBytes;
                    // meta_buffers 存储的thread间隔
                    constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                    // 记录S1~S3的编码字节数
                    int encoded_bytes[kNumStages];

                    // Prefetch: iter0执行S1
                    logfmt_load_global2lds(0, cpy_src_int4_ptr, get_num_logfmt_bytes(0));
                    syncwarp();

                    // Prefetch: iter0执行S2， iter1执行S1
                    if (kNumStages > 2 && kNumIters > 1) {
                        int warp_offset = /*1 * */kNumMsgInt4ElemPerWarp;
                        logfmt_load_global2lds(1, cpy_src_int4_ptr + warp_offset, get_num_logfmt_bytes(warp_offset));

                        int thread_offset = /*0 + */lane_id * kNumSendUnrolls;
                        int num_bytes = logfmt_encode<kNumSendUnrolls>(
                            logfmt_buffers[0],
                            (thread_offset % kNumInt4PerDivision == 0) ? meta_buffers + thread_offset / kNumInt4PerDivision : nullptr,
                            lane_id
                        );
                        encoded_bytes[0] = num_bytes;
                    }
                    syncwarp();

                    // 采用3级流水
                    for (int iter_idx = 0; iter_idx < kNumIters; ++iter_idx) {
                        // 流水线S1: 加载第 (kNumStages-1) 轮之后的数据
                        const int stage_last_iter = iter_idx + kNumStages - 1;  // 当前iter所在stage中的最后一个，初始为S3的读取数据
                        if (stage_last_iter < kNumIters) {
                            int stage_idx = stage_last_iter % kNumStages;
                            int warp_offset = stage_last_iter * kNumMsgInt4ElemPerWarp;
                            logfmt_load_global2lds(stage_idx, cpy_src_int4_ptr + warp_offset, get_num_logfmt_bytes(warp_offset));
                        }

                        // 流水线S2: 处理下一轮的数据量化
                        const int stage_next_iter = iter_idx + 1;
                        if (stage_next_iter < kNumIters) {
                            int stage_idx = stage_next_iter % kNumStages;
                            int warp_offset = stage_next_iter * kNumMsgInt4ElemPerWarp;
                            int thread_offset = warp_offset + lane_id * kNumSendUnrolls;
                            int num_bytes = logfmt_encode<kNumSendUnrolls>(
                                logfmt_buffers[stage_idx],
                                (thread_offset % kNumInt4PerDivision == 0) ? meta_buffers + thread_offset / kNumInt4PerDivision : nullptr,
                                lane_id
                            );
                            encoded_bytes[stage_idx] = num_bytes;
                        }

                        // 流水线S3:当前轮进行数据拷贝到通信显存
                        if (iter_idx < kNumIters) {
                            int stage_idx = iter_idx % kNumStages;
                            using vec_type = uint64_t;
                            int nvecs = encoded_bytes[stage_idx] / sizeof(vec_type);
                            if (nvecs > 0) {
                                UNROLLED_WARP_COPY_LL(1, lane_id, nvecs,
                                    reinterpret_cast<vec_type*>(reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + logfmt_offset_bytes),
                                    reinterpret_cast<vec_type*>(logfmt_buffers[stage_idx]),
                                    ld_direct_global, st_na_global);
                            }
                            logfmt_offset_bytes += encoded_bytes[stage_idx];
                        }

                        syncwarp();
                    }

                    num_send_bytes = logfmt_offset_bytes;

                    // Store metadata
                    using meta_vec_type = uint32_t;
                    UNROLLED_WARP_COPY_LL(1, lane_id, kNumMetaBytes / sizeof(meta_vec_type),
                        reinterpret_cast<meta_vec_type*>(cpy_dst_int4_ptr),
                        reinterpret_cast<meta_vec_type*>(meta_buffers),
                        ld_direct_global, st_na_global);

                } else {
                    // ===== 非 LogFMT 路径：直接 global -> global，不经过 LDS =====
                    for (int iter_idx = 0; iter_idx < kNumIters; ++iter_idx) {
                        int warp_offset = iter_idx * kNumMsgInt4ElemPerWarp;
                        UNROLLED_WARP_COPY_LL(kNumSendUnrolls, lane_id, kNumMsgInt4ElemPerWarp,
                            cpy_dst_int4_ptr + warp_offset,
                            cpy_src_int4_ptr + warp_offset,
                            ld_direct_global, st_na_global);
                        syncwarp();
                    }
                    // 非 LogFMT 时，发送字节数为原始大小
                    num_send_bytes = hidden_bf16_int4 * sizeof(int4); // 或根据实际计算
                }

                syncwarp();
            }

            if (dst_p2p_ptr == 0) {
                internode_ll_putmem_nbi((void*)dst_ptr, (void*)buf_ptr,
                                        num_ranks, dst_rank, local_expert_idx,
                                        num_send_bytes);
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

    // 计算需要多少个warp
    constexpr int num_decode_warps = hidden_bf16_int4 / (kNumRecvUnrolls * kWarpSize);

    // 每128个数据记录一个max/min值，即该数为总的max/min值数量
    constexpr int kNumDivisionBytes = kNumDivisions * sizeof(float);
    // 每个warp内总的BF16值的数量
    constexpr int kNumBF16PerWarpBytes = kWarpSize * kNumRecvUnrolls * sizeof(int4);
    constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes * 10 / 16;

    // 用于记录 max/min 值的 log 值
    auto log_amax_buffers =
        PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_buffer + i * kNumDivisionBytes); });
    auto log_amin_buffers = PatternVisitor([=](const int& i) {
      return reinterpret_cast<float*>(smem_buffer + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes);
    });
    auto cast_info_buffers = PatternVisitor([=](const int& i) {
      return reinterpret_cast<int*>(smem_buffer + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes);
    });

    // 初始化 topk_idx 和 topk_weights
    int topk_idx_by_lane = -1;
    float topk_weights_by_lane = -1;
    int stage_idx = 0;
    for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
        if (lane_id < num_topk) {
            topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
            topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
        }

        for (int w_i = warp_id; w_i < num_decode_warps; w_i += num_warps) {
            float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                int topk_idx_reg = shfl_sync(topk_idx_by_lane, i);
                if (topk_idx_reg < 0)
                    continue;
                const auto& topk_weight_reg = shfl_sync(topk_weights_by_lane, i);

                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const uint8_t*>(reinterpret_cast<uint8_t*>(rdma_recv_x) +
                    (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);

                if constexpr(bSupportLogFMT) {
                    // 接收到的数据位置
                    const uint8_t* data_buffer = rdma_buffer_type + kNumMetaBytes;

                    // 读取max/min数据
                    if(w_i == 0) {
                        // 因为每个warp能处理数据量为 kWarpSize*sizeof(int4)/sizeof(bfloat16) * kNumSendUnrolls
                        // 即不考虑kNumSendUnrolls，一共 kWarpSize*sizeof(int4)/sizeof(bfloat16)/128 组， 代入参数 = kWarpSize / 16 个warp，nv上为2，dcu上为4
                        logfmt_check_amaxmin<kNumDivisions / (kWarpSize / 16), kNumSendUnrolls, kNumRecvUnrolls>(
                            /*meta_buffer*/rdma_buffer_type,
                            reinterpret_cast<int4*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<int4*>(log_amin_buffers[stage_idx]),
                            cast_info_buffers[stage_idx],
                            lane_id);
                    }

                    __syncthreads();

                    // 获取cast_info_buffers
                    const auto& info = cast_info_buffers[stage_idx][w_i];
                    bool enable_cast = info & 1;
                    int num_casted_prefix = info >> 1; // 可用的

                    // 计算偏移（与TMA版本逻辑一致）
                    int warp_offset = kNumLogFMTPerWarpBytes * num_casted_prefix +
                                      kNumBF16PerWarpBytes * (w_i - num_casted_prefix);
                    int lane_offset = (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / kWarpSize * lane_id;

                    // 使用临时缓冲区进行归约
                    const uint8_t* thread_data_ptr = data_buffer + warp_offset + lane_offset;

                    /**
                    一共有kNumDivisions个max/min数据对，读取时每warp默认处理256bit的max/min，所以logfmt_check_amaxmin的kNumLanes设置为 kNumDivisions/2
                    保存数据时每个log_amax_buffers为float2数据类型，保存总的warpkNumDivisions / 2
                    实际保存数据时，每个warp保存的实际数据个数为 kWarpSize*kNumRecvUnrolls*sizeof(int4)/sizeof(hip_bfloat16)
                    实际每个warp读取的max/min的 warp_idx=kWarpSize*kNumRecvUnrolls*sizeof(int4)/sizeof(hip_bfloat16) / 128 = kNumRecvUnrolls * 2
                    具体的lane_id处理的数据量为 warp_idx / kWarpSize
                    */
                    int log_amaxmin_per_warp = kNumRecvUnrolls * kWarpSize * sizeof(int4) / sizeof(hip_bfloat16) / QUANTIZATION_GROUPSIZE;
                    int division_idx = w_i * log_amaxmin_per_warp + lane_id * log_amaxmin_per_warp / kWarpSize;

                    // 反量化
                    decode_and_accumulate<kNumRecvUnrolls>(
                        reinterpret_cast<const uint32_t*>(thread_data_ptr),  // 直接使用全局内存地址
                        combined_values,
                        log_amax_buffers[stage_idx][division_idx],
                        log_amin_buffers[stage_idx][division_idx],
                        enable_cast,
                        topk_weight_reg);
                } else {
                    // 接收到的数据位置
                    const uint8_t* data_buffer = rdma_buffer_type;

                    // 计算偏移
                    int warp_offset = kNumBF16PerWarpBytes * w_i;
                    int lane_offset = kNumBF16PerWarpBytes / kWarpSize * lane_id;
                    // 使用临时缓冲区进行归约
                    const uint8_t* thread_data_ptr = data_buffer + warp_offset + lane_offset;

                    #pragma unroll
                    for (int j = 0; j < kNumRecvUnrolls; ++j) {
                        auto tmp_rdma_value = ld_nc_global(reinterpret_cast<const int4*>(thread_data_ptr) + j);
                        const auto x_bf16 = reinterpret_cast<const hip_bfloat16*>(&tmp_rdma_value);

                        #pragma unroll
                        for (int k = 0; k < kNumElemsPerInt4; ++k) {
                            int combined_idx = j * kNumElemsPerInt4 + k;
                            combined_values[combined_idx] += static_cast<float>(x_bf16[k]) * topk_weight_reg;
                        }
                    }
                }
            }

            // Write results，kNumRecvUnrolls==2时则写256bit的数
            int4 combined_int4[kNumRecvUnrolls];
            auto combined_bf16 = reinterpret_cast<hip_bfloat16 *>(&combined_int4[0]);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4 * kNumRecvUnrolls; ++ j) {
                combined_bf16[j] = static_cast<hip_bfloat16>(combined_values[j]);
            }

            for(int j = 0; j < kNumRecvUnrolls; ++ j) {
                (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 +
                w_i * kWarpSize * kNumRecvUnrolls)[lane_id * kNumRecvUnrolls + j] = combined_int4[j];
            }
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
             bool use_logfmt,
             void* workspace, int num_device_sms, hipStream_t stream,
             int phases, bool zero_copy) {
    constexpr int kMaxNumWarps = 8;
    constexpr int kNumMaxTopk = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = kMaxNumWarps / num_warp_groups; // num_warps_per_group>1, "Requires more than one warp per group"
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden)                                            \
  {                                                                            \
    auto combine_func = use_logfmt ?                                           \
         combine<true, hidden, kNumMaxTopk, kMaxNumWarps> :                    \
         combine<false, hidden, kNumMaxTopk, kMaxNumWarps>;                    \
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, combine_func,                          \
        combined_x, rdma_recv_x, rdma_recv_flag, rdma_send_x,                  \
        x, topk_idx, topk_weights, src_info, layout_range,                     \
        global_atomic_counter, combine_wait_recv_cost_stats,                   \
        next_clean, num_next_clean_int,                                        \
        atomic_clean_flag, num_combined_tokens, hidden,                        \
        num_topk, num_max_dispatch_tokens_per_rank,                            \
        num_experts, rank, num_ranks,                                          \
        num_warp_groups, num_warps_per_group, phases, zero_copy);              \
  }                                                                            \
  break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

template <int kHidden, int kQuantType=0, int kQuantGroupSize=0, int kMaxNumWarps=16>
__global__ __launch_bounds__(16 * kWarpSize, 1) void
    dispatch_ll_layered(
             bool disable_ll_layered,
             void* packed_recv_x, void* packed_recv_x_scales,
             int64_t* packed_recv_src_info, int64_t* packed_recv_layout_range,
             int* packed_recv_count,
             int* global_atomic_counter,
             void* rdma_recv_x, int64_t* rdma_recv_count, void* rdma_x,
             const void* x, const int64_t* topk_idx,
             int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
             int64_t* next_clean, int num_next_clean_int,
             int num_tokens, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             int num_warp_groups, int num_warps_per_group,
             bool fp8_round_scale, int phases) {
    // 定义量化类型的枚举
    enum class QuantType {
        None        = 0,        // 不进行量化
        Int8        = 1,        // 采用 Int8 量化
        FP8_E4M3    = 2,        // 采用 FP8 量化 __HIP_E4M3
        FP8_UE8M0   = 3,        // 采用 FP8 量化 DeepseekV3.1的 UE8M0
        FP8_E5M2    = 4         // 采用 FP8 量化 __HIP_E5M2
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / kWarpSize, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    char* rdma_recv_x_cahr_ptr = reinterpret_cast<char*>(rdma_recv_x);

    const auto num_nvl_ranks = NUM_MAX_NVL_PEERS;
    const auto num_nodes = num_ranks / num_nvl_ranks;

    int* data_ready_counter = reinterpret_cast<int*>(rdma_recv_count + num_experts);
    int* data_ready_send_buffer =
        data_ready_counter + num_nodes * num_max_dispatch_tokens_per_rank * num_nvl_ranks;

    int* next_clean_data_ready_counter = reinterpret_cast<int*>(next_clean + num_experts);
    
    if (!disable_ll_layered) {
        if (thread_id < num_nvl_ranks) {
            __hip_atomic_store(data_ready_send_buffer + thread_id, 2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    __syncthreads();

    // May extract UE8M0 from the scales
    constexpr bool kUseQuant8Bit = kQuantType > 0;
    constexpr bool kUseUE8M0 = kQuantType == 3; // QuantType::FP8_UE8M0
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = QUANTIZATION_GROUPSIZE;
    constexpr int kNumScales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseQuant8Bit ? sizeof(__hip_fp8_storage_t) : sizeof(hip_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseQuant8Bit, int2, int4>::type;

    const size_t num_bytes_per_meta = sizeof(int4);
    const size_t num_bytes_per_data = (kUseQuant8Bit ? (kHidden + (kQuantGroupSize == 0 ? 4 : kNumScales) * sizeof(float)) : (kHidden * sizeof(hip_bfloat16)));
    const size_t num_bytes_per_msg = num_bytes_per_meta + num_bytes_per_data;
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    char* rdma_recv_x_meta = rdma_recv_x_cahr_ptr;
    char* rdma_recv_x_data = rdma_recv_x_cahr_ptr + num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_meta;

    // Expert counts
    __shared__ int shared_num_tokens_sent_per_expert[kMaxNumWarps];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(hip_bfloat16);
        constexpr int kNumThreadPerGroup = QUANTIZATION_GROUPSIZE / kNumElemsPerRead;
        // EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = num_warps * kWarpSize;
        constexpr int hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index write
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // 用于记录per-channel量化的amax
            __shared__ float channel_amaxf[kNumScales];
            if constexpr(kUseQuant8Bit && kQuantGroupSize == 0) {
                if (thread_id < kNumScales) {
                    channel_amaxf[thread_id] = 0.0;
                }
                __syncthreads();
            }

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr(kUseQuant8Bit) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<hip_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = 0.0, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }
                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize / kNumPerChannels == 4, "Invalid vectorization");
                    amax = warp_reduce_max<kNumThreadPerGroup>(amax);
                    const int scale_offset = i * kNumElemsPerRead / QUANTIZATION_GROUPSIZE;

                    if constexpr(kQuantGroupSize == 0) {
                        // 记录每128个数的最大值
                        channel_amaxf[scale_offset] = fmaxf(amax, channel_amaxf[scale_offset]);
                    } else {
                        calculate_quant8bit_scales<kQuantType>(amax, scale, scale_inv, fp8_round_scale);
                        if (lane_id % kNumThreadPerGroup == 0)
                            rdma_x_scales[scale_offset] = scale_inv;

                        // Cast into send buffer
                        vec_t int2_value;
                        pack_quantized_values<kQuantType, kNumElemsPerRead>(fp32_values, scale, int2_value);
                        rdma_x_vec[i] = int2_value;
                    }
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            __syncthreads();

            if constexpr(kUseQuant8Bit && kQuantGroupSize == 0) {
                float amax_per_token = 0.0;
                // 并行规约，计算每个token的amax
                for (int s = 0; s < kNumScales; s+=kWarpSize) {
                    int src_idx = s + lane_id;
                    float tmp_amaxf = 0;
                    if(src_idx < kNumScales) {
                        tmp_amaxf = channel_amaxf[src_idx];
                    }
                    tmp_amaxf = warp_reduce_max<kWarpSize>(tmp_amaxf);
                    channel_amaxf[0] = fmaxf(tmp_amaxf, channel_amaxf[0]);
                    __syncthreads();
                }
                amax_per_token = channel_amaxf[0];

                // 根据最大值计算scale
                float scale, scale_inv;
                calculate_quant8bit_scales<kQuantType>(amax_per_token, scale, scale_inv, fp8_round_scale);
                if (thread_id == 0) {
                    rdma_x_scales[0] = scale_inv;
                }

                for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                    // Read
                    auto int4_value = __ldg(x_int4 + i);
                    auto bf16_values = reinterpret_cast<hip_bfloat16*>(&int4_value);

                    // Cast into send buffer
                    vec_t int2_value;
                    pack_quantized_values<kQuantType, kNumElemsPerRead>(bf16_values, scale, int2_value);
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
                if(!disable_ll_layered){
                    int send_node_id = dst_expert_idx / num_local_experts / num_nvl_ranks;
                    auto real_write_dst_rank = dst_rank / num_nvl_ranks * num_nvl_ranks +
                        rank % num_nvl_ranks;  // send data to same gpu_device_id_rank(same-rail rdma traffic)

                    auto real_dst_expert_id = real_write_dst_rank * num_local_experts + dst_expert_local_idx;

                    auto tmp_dst_expert_id = lane_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id)) : -1;
                    auto tmp_dst_node_id = tmp_dst_expert_id >= 0 ? tmp_dst_expert_id / num_local_experts / num_nvl_ranks : -1;

                    for (int i = 0; i < warp_id; ++i) {
                        auto dst_node_id = shfl_sync(tmp_dst_node_id, i);  // broadcast
                        if (dst_node_id == send_node_id) {                 // whether to send repeatedly
                            send_node_id = -1;
                            break;
                        }
                    }

                    if (send_node_id != -1) {
                    // =======================================  token data ==========================================
                        int* src_data_ptr = rdma_x_src_idx + 4;
                        char* dst_data_ptr = rdma_recv_x_data +
                                (rank / num_nvl_ranks) * num_max_dispatch_tokens_per_rank * num_bytes_per_data +
                                token_idx * num_bytes_per_data;
                        const auto p2p_data_ptr = internode::shmem_get_p2p_ptr((void*)(dst_data_ptr), rank, real_write_dst_rank);
                        if (!p2p_data_ptr) {
                            internode_ll_putmem_nbi(
                                reinterpret_cast<void*>(dst_data_ptr), reinterpret_cast<void*>(src_data_ptr),
                                num_ranks, real_write_dst_rank, dst_expert_local_idx, num_bytes_per_data);  
                        } else {    
                            const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_data_ptr);
                            const auto* dst_int4_ptr = reinterpret_cast<int4*>(p2p_data_ptr);
                            UNROLLED_WARP_COPY_LL(8, lane_id, num_bytes_per_data / sizeof(int4), dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                        }
                        // ========================================  token data flag =======================================
                        uint64_t src_data_flag_ptr = reinterpret_cast<uint64_t>(data_ready_send_buffer);
                        const auto data_ready_counter_ptr = reinterpret_cast<uint64_t>(data_ready_counter) +
                                                        (rank / num_nvl_ranks) * num_max_dispatch_tokens_per_rank * num_nvl_ranks * sizeof(int) +
                                                        token_idx * num_nvl_ranks * sizeof(int);

                        uint64_t data_ready_counter_p2p_ptr = internode::shmem_get_p2p_ptr((void*)(data_ready_counter_ptr), rank, real_write_dst_rank);
                        if (data_ready_counter_p2p_ptr == 0) {
                            // internode::shmemx_int8_put_nbi_warp_refactoring(
                            //     reinterpret_cast<signed char*>(data_ready_counter_ptr), reinterpret_cast<signed char*>(src_data_flag_ptr), 
                            //     num_nvl_ranks * sizeof(int), num_ranks + dst_expert_local_idx * num_ranks + real_write_dst_rank, rank, real_write_dst_rank, true);  
                            internode_ll_putmem_nbi(
                                reinterpret_cast<void*>(data_ready_counter_ptr), reinterpret_cast<void*>(src_data_flag_ptr),
                                num_ranks, real_write_dst_rank, dst_expert_local_idx, num_nvl_ranks * sizeof(int));  
                        } else {
                            int* dst_int_ptr = reinterpret_cast<int*>(data_ready_counter_p2p_ptr);
                            if(lane_id < num_nvl_ranks){
                                __hip_atomic_store(dst_int_ptr + lane_id, 2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                            }
                        }
                    }
                    // =========================  meta data=============================
                    const auto src_meta_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                    const auto dst_meta_ptr = reinterpret_cast<uint64_t>(rdma_recv_x_meta) +
                            dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_meta +
                            rank * num_max_dispatch_tokens_per_rank * num_bytes_per_meta +
                            slot_idx * num_bytes_per_meta;
        
                    uint64_t p2p_meta_ptr = internode::shmem_get_p2p_ptr((void*)(dst_meta_ptr), rank, dst_rank); 
                    if (!p2p_meta_ptr) {
                        // internode::shmemx_int8_put_nbi_warp_refactoring(
                        //     reinterpret_cast<signed char*>(dst_meta_ptr), reinterpret_cast<signed char*>(src_meta_ptr),
                        //     num_bytes_per_meta, num_ranks + dst_expert_local_idx * num_ranks + dst_rank, rank, dst_rank, true);  
                        internode_ll_putmem_nbi(
                            reinterpret_cast<void*>(dst_meta_ptr), reinterpret_cast<void*>(src_meta_ptr),
                            num_ranks, dst_rank, dst_expert_local_idx, num_bytes_per_meta);  
                    } else {    
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_meta_ptr);
                        int4* dst_int4_ptr = reinterpret_cast<int4*>(p2p_meta_ptr);
                        if(lane_id==0){
                            dst_int4_ptr[0] = src_int4_ptr[0];
                        }
                    }
                    
                    syncwarp();
                    lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
                    lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + real_dst_expert_id, 1) : 0;
                } else {
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
    }
    if (warp_id == num_warps - 1) {
        // EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            if (disable_ll_layered) {
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
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kMaxNumWarps] = {0};
        int waiting_flag[kMaxNumWarps] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += kWarpSize) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
            if (!disable_ll_layered) {
                if (idx < 0)
                    continue;
                const auto dst_rank = idx / num_local_experts;
                const auto dst_expert_local_idx = idx % num_local_experts;
                auto real_write_dst_rank = dst_rank / num_nvl_ranks * num_nvl_ranks + rank % num_nvl_ranks;
                auto real_dst_expert_id = real_write_dst_rank * num_local_experts + dst_expert_local_idx;
                if (real_dst_expert_id >= expert_begin_idx and real_dst_expert_id < expert_end_idx)
                    waiting_flag[real_dst_expert_id - expert_begin_idx] ++;
            }
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            auto waiting_flag_sum = 0;
            if (!disable_ll_layered) {  // only open ll dispatch opt, should do
                waiting_flag_sum = warp_reduce_sum(waiting_flag[i - expert_begin_idx]);
            }
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - waiting_flag_sum - sum);
            }
        }
    }

    if (!disable_ll_layered and sm_id == num_sms - 1) {
        // The first SM is also responsible for cleaning the next buffer
        for (int i = thread_id; i < num_experts; i += blockDim.x)  // clean for combine
            next_clean[i] = 0;

        // clean data ready flag 
        for (int i = thread_id; i < num_max_dispatch_tokens_per_rank * num_ranks; i += blockDim.x) {
            int token_idx = i / num_ranks;
            int rank_id = i % num_ranks;

            auto node_id = rank_id / num_nvl_ranks;
            auto nvl_rank_id = rank_id % num_nvl_ranks;

            auto* data_ready_flag_ptr = reinterpret_cast<int*>(next_clean_data_ready_counter) +
                node_id * num_max_dispatch_tokens_per_rank * num_nvl_ranks + token_idx * num_nvl_ranks + rank % num_nvl_ranks;
            EP_DEVICE_ASSERT(data_ready_flag_ptr - next_clean_data_ready_counter <
                                num_max_dispatch_tokens_per_rank * num_nodes * num_nvl_ranks * sizeof(int));
            const auto data_ready_p2p_src_ptr =
                internode::shmem_get_p2p_ptr((void*)(data_ready_flag_ptr), rank, rank / num_nvl_ranks * num_nvl_ranks + nvl_rank_id); 

            reinterpret_cast<int*>(data_ready_p2p_src_ptr)[0] = 0;
        }

        __syncthreads();
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += blockDim.x)
            atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
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
        uint8_t* rdma_recv_x_uint8 = nullptr;
        if (!disable_ll_layered) {
            rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x_meta) +
                                    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_meta +
                                    src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_meta;
        }
        if (disable_ll_layered) {
            rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
                                        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                        src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        }
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                                 local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = ALIGN<int>(kNumScales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
                                   local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank *
                                       (kQuantGroupSize == 0 ? 1 : num_aligned_scales);

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kMaxNumWarps], shared_recv_token_begin_idx[kMaxNumWarps];

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
        const auto real_read_src_rank = src_rank % num_nvl_ranks + rank / num_nvl_ranks * num_nvl_ranks;
        // Copy tokens
        EP_STATIC_ASSERT(kNumScales <= 64, "Invalid hidden size");
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            int4* src_data = nullptr;
            if (!disable_ll_layered) {
                int* src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_meta);
                int src_token_idx = __builtin_nontemporal_load(src_src_idx);
                if (lane_id == 0) {
                    recv_src_info[recv_token_begin_idx + i] = pack2<int, int64_t>(src_token_idx, src_rank);
                }
                            
                const auto data_ready_flag_src_ptr = data_ready_counter +
                    (src_rank / num_nvl_ranks) * num_max_dispatch_tokens_per_rank * num_nvl_ranks + 
                    src_token_idx * num_nvl_ranks + 
                    rank % num_nvl_ranks;
                const auto src_data_ready_flag_p2p_ptr =
                    reinterpret_cast<int*>(internode::shmem_get_p2p_ptr((void*)(data_ready_flag_src_ptr), rank, real_read_src_rank)); 
                if (lane_id == 0) {
                    int tmp = 0;
                    auto start_time = clock64();
                    bool flag_get = false;
                    while (tmp != 2) {
                        tmp = __hip_atomic_load(src_data_ready_flag_p2p_ptr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
                        if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                            printf(
                                "DeepEP ll dispatch recv data timeout, src_rank:%d, dst_rank: %d, real_read_src_rank:%d,src_token_idx:%d "
                                "dst RDMA lane: %d, num_recv_tokens: %d\n",
                                src_rank,
                                rank,
                                real_read_src_rank,
                                src_token_idx,
                                lane_id,
                                num_recv_tokens
                                );
                            break;
                        }
                    }
                }

                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_recv_x_data) +
                    (src_rank / num_nvl_ranks) * num_max_dispatch_tokens_per_rank * num_bytes_per_data 
                    + src_token_idx * num_bytes_per_data;

                uint64_t src_ptr_p2p = internode::shmem_get_p2p_ptr((void*)(src_ptr), rank, real_read_src_rank);

                src_data = reinterpret_cast<int4*>(src_ptr_p2p);
            }
            if (disable_ll_layered) {
                const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
                int src_token_idx = __builtin_nontemporal_load(src_src_idx);
                if (lane_id == 0)
                    // 加入 源rank 信息
                    recv_src_info[recv_token_begin_idx + i] = pack2<int, int64_t>(src_token_idx, src_rank);
                syncwarp();

                // Copy data
                // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
                src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            }
            
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY_LL(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr(kUseQuant8Bit) {
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;

                if constexpr(kQuantGroupSize == 0) {
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

void dispatch_ll_layered(bool dispatch_ll_dispatch_opt,
                         void* packed_recv_x, void* packed_recv_x_scales,
                         int64_t* packed_recv_src_info, int64_t* packed_recv_layout_range,
                         int* packed_recv_count,
                         int* global_atomic_counter,
                         void* rdma_recv_x, int64_t* rdma_recv_count, void* rdma_x,
                         const void* x, const int64_t* topk_idx,
                         int64_t* next_clean, int num_next_clean_int,
                         int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
                         int num_topk, int num_experts, int rank, int num_ranks,
                         int quant_type, int quant_group_size, bool fp8_round_scale,
                         void* workspace, int num_device_sms,
                         hipStream_t stream, int phases) {
    constexpr int kMaxNumWarps = 16;
    constexpr int kNumMaxTopK = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = kMaxNumWarps / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // 限制groupsize的大小
    EP_HOST_ASSERT(quant_group_size == 0 || quant_group_size == 128);

    /*量化类型枚举
    0 -> None          不量化，保持原始精度
    1 -> Int8          使用 INT8 对称量化
    2 -> FP8_E4M3      使用 FP8 E4M3 格式 (__HIP_E4M3)
    3 -> FP8_UE8M0     使用 DeepSeekV3.1 提出的 UE8M0 格式 (仅支持round_scale=True)
    4 -> FP8_E5M2      使用 FP8 E5M2 格式 (__HIP_E5M2)
    */

#define DISPATCH_LL_LAUNCH_CASE(hidden)                                                        \
  {                                                                                            \
    auto dispatch_func = dispatch_ll_layered<hidden, 0, 0, kMaxNumWarps>;                      \
    if (quant_group_size == 0) {                                                               \
        switch (quant_type) {                                                                  \
            case 1: dispatch_func = dispatch_ll_layered<hidden, 1, 0, kMaxNumWarps>; break;    \
            case 2: dispatch_func = dispatch_ll_layered<hidden, 2, 0, kMaxNumWarps>; break;    \
            case 3: dispatch_func = dispatch_ll_layered<hidden, 3, 0, kMaxNumWarps>; break;    \
            case 4: dispatch_func = dispatch_ll_layered<hidden, 4, 0, kMaxNumWarps>; break;    \
        }                                                                                      \
    } else {                                                                                   \
        switch (quant_type) {                                                                  \
            case 1: dispatch_func = dispatch_ll_layered<hidden, 1, 128, kMaxNumWarps>; break;  \
            case 2: dispatch_func = dispatch_ll_layered<hidden, 2, 128, kMaxNumWarps>; break;  \
            case 3: dispatch_func = dispatch_ll_layered<hidden, 3, 128, kMaxNumWarps>; break;  \
            case 4: dispatch_func = dispatch_ll_layered<hidden, 4, 128, kMaxNumWarps>; break;  \
        }                                                                                      \
    }                                                                                          \
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, dispatch_func, dispatch_ll_dispatch_opt,               \
        packed_recv_x, packed_recv_x_scales,                                                   \
        packed_recv_src_info, packed_recv_layout_range, packed_recv_count,                     \
        global_atomic_counter,                                                                 \
        rdma_recv_x, rdma_recv_count, rdma_x, x, topk_idx,                                     \
        atomic_counter_per_expert, atomic_finish_counter_per_expert,                           \
        next_clean, num_next_clean_int,                                                        \
        num_tokens, num_max_dispatch_tokens_per_rank,                                          \
        num_topk, num_experts, rank, num_ranks,                                                \
        num_warp_groups, num_warps_per_group, fp8_round_scale, phases);                        \
  }                                                                                            \
  break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(DISPATCH_LL_LAUNCH_CASE);
#undef DISPATCH_LL_LAUNCH_CASE
}

/*
    combine 启用 overlop 后的实现
*/ 
template <int kHidden, int kNumMaxTopk, int kMaxNumWarps=16>
__global__ __launch_bounds__(16 * kWarpSize, 1) void
combine_sbo(bool disable_ll_layered,
        void* combined_x,
        void* rdma_recv_x, int64_t* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int64_t* src_info, const int64_t* layout_range,
        // Overlap specific parameters
        int* packed_recv_count, int* comp_signal, int block_m, int threshold,
        int* global_atomic_counter,
        int64_t* combine_wait_recv_cost_stats,
        int64_t* next_clean, int num_next_clean_int,
        int* atomic_clean_flag, int* atomic_finish_counter_per_expert,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy) {
    // 假设 启用 3 个block
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / kWarpSize, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;    // 16
    const auto warp_group_id = warp_id / num_warps_per_group;  // 0 0 0 ...  0
    const auto sub_warp_id = warp_id % num_warps_per_group;    // 0 1 2 ...  15
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;  // 这意味着 一次 并行处理 3个专家  0 1 2
    
    int* next_clean_data_ready_counter = reinterpret_cast<int*>(next_clean + num_experts);
    const auto num_nvl_ranks = NUM_MAX_NVL_PEERS;
    const auto num_nodes = num_ranks / num_nvl_ranks;

    // hidden_bf16_int4： bf16 的 token 包含多少个 int4
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(hip_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    EP_STATIC_ASSERT(kHidden % QUANTIZATION_GROUPSIZE == 0, "Invalid hidden");
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(hip_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Shared between warps in sms for overlap mode, where each sm only has one warp group
    __shared__ volatile int shared_vaild_signal_prefix_sum[40];  // 用于统计 本地专家 有效信号 的 前缀和

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    if (!disable_ll_layered and sm_id == num_sms - 1) {
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            next_clean[i] = 0;

        // clean data ready flag
        for (int i = thread_id; i < num_max_dispatch_tokens_per_rank * num_ranks; i += num_threads) {
            int token_idx = i / num_ranks;
            int rank_id = i % num_ranks;
            {
                auto node_id = rank_id / num_nvl_ranks;
                auto nvl_rank_id = rank_id % num_nvl_ranks;
                auto* data_ready_flag_ptr = reinterpret_cast<int*>(next_clean_data_ready_counter) +
                    node_id * num_max_dispatch_tokens_per_rank * num_nvl_ranks + token_idx * num_nvl_ranks + rank % num_nvl_ranks;
                EP_DEVICE_ASSERT(data_ready_flag_ptr - next_clean_data_ready_counter <
                                 num_max_dispatch_tokens_per_rank * num_nodes * num_nvl_ranks * sizeof(int));
                const auto data_ready_p2p_src_ptr =
                    internode::shmem_get_p2p_ptr((void*)(data_ready_flag_ptr), rank, rank / num_nvl_ranks * num_nvl_ranks + nvl_rank_id);
                reinterpret_cast<int*>(data_ready_p2p_src_ptr)[0] = 0;
            }
        }
        // Notify before executing `int_p`
        __syncthreads();
        if (thread_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    if (disable_ll_layered) {
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
    }

    __syncthreads();

    // ========================================
    //  shared_vaild_signal_sum： 本地专家的总信号量
    //  shared_local_expert_idx： 共享内存中的 本地专家索引。初始置为 0 , 表明 当前 block 当前在 处理的 本地专家索引
    __shared__ int shared_vaild_signal_sum, shared_local_expert_idx;

    // 计算每个 本地专家 有效信号 计数 的 前缀和，即使没有 token, 也算作一个 任务
    if (sub_warp_id == 0 and lane_id == 0) { // 0号 warp 的 0号线程 执行下述操作
        shared_vaild_signal_prefix_sum[0] = (packed_recv_count[0] == 0 ? 1 : ceil_div(packed_recv_count[0], block_m));
        shared_local_expert_idx = 0;  // 共享内存中 本地专家索引 置为 0 

        for (int i = 1; i < num_local_experts; i++) {
            shared_vaild_signal_prefix_sum[i] =
                shared_vaild_signal_prefix_sum[i - 1] + (packed_recv_count[i] == 0 ? 1 : ceil_div(packed_recv_count[i], block_m));
        }

        shared_vaild_signal_sum = shared_vaild_signal_prefix_sum[num_local_experts - 1];
    }
    __syncthreads();  // 等待前缀和 统计完成 16个 warp 同步等待

    // 每个 block 负责一个 处理信号，并循环处理到 最后
    for (int vaild_signal_idx = sm_id; vaild_signal_idx < shared_vaild_signal_sum; vaild_signal_idx += num_sms) {
        // ======================  16个 warp 进入  ======================

        // 通过扫描前缀和数组找到当前处理的本地专家索引，并记录在 shared_local_expert_idx
        if (sub_warp_id == 0 and lane_id == 0) {
            while (vaild_signal_idx >= shared_vaild_signal_prefix_sum[shared_local_expert_idx])
                shared_local_expert_idx++;
        }
        __syncthreads();

        // ===========================================
        // shared_local_expert_idx： 当前处理的任务块 是哪个本地专家
        // 上述 操作 确定了  当前 block 负责处理的本地专家为 shared_local_expert_idx
        // 需要依据 shared_local_expert_idx 本地索引确定其他 地址

        const auto local_expert_idx = shared_local_expert_idx;  // 当前处理 的 本地专家索引
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;  // 获取 本地专家 在全局中的索引
        const auto local_x = static_cast<const int4*>(x) + 
                                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) + 
                                        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // ================================  等待相应的 comp_signal 达到阈值
        //----------------------- 确定 当前等待的信号量位置 
        // num_tokens_per_expert：当前 负责的专家 dispatch 阶段 接收的 总 token 数
        // num_signal_per_expert：当前 负责的专家 需要等待的总 信号 数
        // local_expert_signal_idx： 当前处理的信号总索引，是 当前处理专家的 第几个信号
        int num_tokens_per_expert, num_signal_per_expert, local_expert_signal_idx;
        const int* gemm_comp_signal;
        num_tokens_per_expert = packed_recv_count[local_expert_idx];  // 当前专家 dispatch 阶段接收的 总 token 数
        num_signal_per_expert = ceil_div(num_ranks * num_max_dispatch_tokens_per_rank, block_m);  // 每个专家的 最大 信号数
        local_expert_signal_idx =
            (local_expert_idx == 0) ? vaild_signal_idx : vaild_signal_idx - shared_vaild_signal_prefix_sum[local_expert_idx - 1];  // 当前专家 中的 信号索引
        gemm_comp_signal = comp_signal + num_signal_per_expert * local_expert_idx + local_expert_signal_idx;

        //----------------------- 循环等待 信号量到达 阈值
        if (sub_warp_id == 0 and lane_id == 0 and num_tokens_per_expert != 0) { // 当前专家 dispatch 阶段接收的 token 数 不是 0 的话，循环等待 信号量的值 到达 阈值
            while (ld_acquire_global(gemm_comp_signal) != threshold)
                ;
        }

        __syncthreads();

        // ============================== 发射 RDMA 指令 ==============================
        // ------------------------------ 确定 处理的 token 起始位置 和 结束位置 -----------------
        auto token_start_idx = local_expert_signal_idx * block_m;
        auto token_end_idx = min((local_expert_signal_idx + 1) * block_m, num_tokens_per_expert);
        // 16个 warp 每个warp 负责一个 token 的发射
        for (int token_idx = sub_warp_id + token_start_idx; token_idx < token_end_idx; token_idx += num_warps_per_group) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);
    
            const auto dst_rank = static_cast<int>(__ldg(local_src_info + token_idx) >> 32);         
            const auto src_idx = static_cast<int>(__ldg(local_src_info + token_idx) & 0xffffffff);

            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
        
            uint64_t p2p_ptr = internode::shmem_get_p2p_ptr((void*)dst_ptr, rank, dst_rank);
            if (p2p_ptr == 0) {  // RDMA
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy){
                    UNROLLED_WARP_COPY_LL(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                }

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

        __syncthreads(); // 等待 16 个 warp 都完成 RDMA 发射 

        // =================================  当前所有 RDMA 下发完成后，判断是不是要 发射 完成的 flag=====================================
        bool put_finish_flag = false;  // 标记是不是要发射 RDMA 结束标记
        // 判断是不是 到了 当前专家处理的 最后
        if (sub_warp_id == 0) {  // 
            if (lane_id == 0) {
                const auto finish_counter = (num_tokens_per_expert == 0 ? 1 : ceil_div(num_tokens_per_expert, block_m));   // 获取当前专家 发送的 总 的信号数
                if ((atomicAdd(atomic_finish_counter_per_expert + local_expert_idx, 1) + 1) == finish_counter)
                    put_finish_flag = true;
            }
            put_finish_flag = shfl_sync(put_finish_flag, 0);
        }

        __syncthreads();
        // 通知其他 所有 rank，当前本地专家的 token 已经发射完成
        if (sub_warp_id == 0 and put_finish_flag) {
            for (int dst_rank = lane_id; dst_rank < num_ranks; dst_rank += 64) {

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
            if (lane_id == 0) // 清理 标记数组
                atomic_finish_counter_per_expert[local_expert_idx] = 0;
        }

        __syncthreads();
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

void combine_sbo(void* combined_x,
                 void* rdma_recv_x, int64_t* rdma_recv_flag, void* rdma_send_x,
                 const void* x, const int64_t* topk_idx, const float* topk_weights,
                 const int64_t* src_info, const int64_t* layout_range,
                 // Overlap 新增控制参数
                 bool disable_ll_layered,
                 int* packed_recv_count, int* comp_signal,
                 int block_m, int threshold, int num_sms,
                 // 同步与统计参数
                 int* global_atomic_counter,
                 int64_t* combine_wait_recv_cost_stats,
                 int64_t* next_clean, int num_next_clean_int,
                 // 维度与配置参数
                 int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
                 int num_topk, int num_experts, int rank, int num_ranks,
                 // 系统资源与执行参数
                 void* workspace, int num_device_sms, hipStream_t stream,
                 int phases, bool zero_copy) {
    constexpr int kMaxNumWarps = 16;
    constexpr int kNumMaxTopk = 11;

    int num_warp_groups, num_warps_per_group, num_recv_per_sm, num_warps;

    if (phases == LOW_LATENCY_SEND_PHASE) {   // 如果启用  overlop 必须是 send 阶段
        num_warp_groups = 1;   // 一个 block 只有一个 warp 组
        num_warps_per_group = 16;   // 16 个 warp 每个 warp 64 线程
        num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
        EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0 and block_m > 0 and threshold > 0);

        num_warps = num_warp_groups * num_warps_per_group;
    } else {
        num_warp_groups = ceil_div(num_experts, num_device_sms);
        num_warps_per_group = kMaxNumWarps / num_warp_groups;
        num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
        EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

        num_warps = num_warp_groups * num_warps_per_group;
        num_sms = max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));
    }

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_clean_flag + 1;  // overlop 新增使用
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_OVERLOP_LAUNCH_CASE(hidden)                                                     \
  {                                                                                             \
    auto combine_overlop_func = combine_sbo<hidden, kNumMaxTopk, kMaxNumWarps>;                 \
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, combine_overlop_func,                                   \
        disable_ll_layered,                                                                     \
        combined_x, rdma_recv_x, rdma_recv_flag, rdma_send_x,                                   \
        x, topk_idx, topk_weights, src_info, layout_range,                                      \
        packed_recv_count, comp_signal, block_m, threshold,                                     \
        global_atomic_counter, combine_wait_recv_cost_stats,                                    \
        next_clean, num_next_clean_int,                                                         \
        atomic_clean_flag, atomic_finish_counter_per_expert,                                    \
        num_combined_tokens, hidden,                                                            \
        num_topk, num_max_dispatch_tokens_per_rank,                                             \
        num_experts, rank, num_ranks,                                                           \
        num_warp_groups, num_warps_per_group, phases, zero_copy);                               \
  }                                                                                             \
  break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(COMBINE_OVERLOP_LAUNCH_CASE);
#undef COMBINE_OVERLOP_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace deep_ep
