#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "buffer.cuh"
#include "utils.cuh"
#include <iostream>

#include "hip/hip_runtime.h"

#include "shmem_wrapper.cuh"

namespace deep_ep {

namespace internode_ll {


template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmt_encode(int4* lds_buffer, __hip_bfloat162* shared_amaxmin, const int& lane_id) {
    EP_STATIC_ASSERT(kNumSendUnrolls == 2, "kNumSendUnrolls == 2 only");
    
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(__hip_bfloat16); // 8
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1); // = 512
    constexpr int kSendValueBytes = kNumSendUnrolls * sizeof(int4); //=2*16=32
    constexpr int kNumElementPerInt4 = sizeof(int4) / sizeof(uint32_t);

    int4 int4_values[kNumSendUnrolls];
    const auto& uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    const auto& bf162_values = reinterpret_cast<__hip_bfloat162*>(int4_values);

    // Calculate lane offset
    const auto& ld_buffer = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(lds_buffer) + lane_id * kSendValueBytes);
    const auto& st_buffer = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(lds_buffer) + lane_id * kSendValueBytes * 10 / 16);

    // Local log amax
    auto bf162_amax = __hip_bfloat162(HIPRT_ZERO_BF16, HIPRT_ZERO_BF16);
    auto bf162_amin = __hip_bfloat162(HIPRT_INF_BF16, HIPRT_INF_BF16);

    uint32_t local_signs = 0;

    #pragma unroll
    for (int v = 0; v < kNumSendUnrolls; ++v) {
        int4 ld_int4_value = ld_nc_global(ld_buffer + v); // 向量化读取
        uint32_t* ld_u32_ptr = reinterpret_cast<uint32_t*>(&ld_int4_value);

        #pragma unroll
        for (int k = 0; k < kNumElementPerInt4; ++k) { // 也是kNumSendUnrolls * kNumElemsPerInt4 / 2
            // TODO: eliminate bank conflicts
            uint32_t ld_u32_value = ld_u32_ptr[k];
            int k_offset = v * kNumElementPerInt4 + k;

            // 提取符号位: 每个bfloat16的最高位是符号位
            local_signs |= ((ld_u32_value >> 15) & 1) << (k_offset * 2);
            local_signs |= ((ld_u32_value >> 31) & 1) << (k_offset * 2 + 1);
            // 清除符号位，保留幅值
            ld_u32_value &= 0x7fff7fff;

            auto ld_bf16_value = *reinterpret_cast<__hip_bfloat162*>(&ld_u32_value);
            bf162_amax = __hmax2(bf162_amax, ld_bf16_value);
            bf162_amin = __hmin2(bf162_amin, ld_bf16_value);

            uint32_values[k_offset] = ld_u32_value;
        }
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    const auto& fp162_max = __bfloat1622float2(bf162_amax);

    auto amax = __builtin_fmaxf(static_cast<float>(bf162_amax.x), static_cast<float>(bf162_amax.y));
    auto amin = __builtin_fminf(static_cast<float>(bf162_amin.x), static_cast<float>(bf162_amin.y));

    // 即每128个值进行一次reduce
    constexpr static int kNumLanesToReduce = 128 * sizeof(__hip_bfloat16) / kSendValueBytes; // =128*2 / (kNumSendUnrolls * sizeof(int4)) = 8
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (shared_amaxmin != nullptr) {
        *shared_amaxmin = __hip_bfloat162(amax, amin);
    }

    // Calculate log amin/amax float
    const auto& log_amax = __builtin_amdgcn_logf(amax);
    const auto& log_amin = __builtin_fmaxf(__builtin_amdgcn_logf(amin), log_amax - kMinClip);

    // 在组内广播enable_cast结果
    const bool& enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        // 计算10bit数据的两个相邻数值的差值
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        const auto step_inv = 1.0f / step;

        // 计算舍入值
        const auto rounding = 2.0f - __builtin_amdgcn_logf((1.0f + __builtin_amdgcn_exp2f(step)) * 0.5f) * step_inv;
        const auto fused_rounding = rounding - log_amin * step_inv;

        // 用于存储编码后的值
        uint32_t encoded[kNumElemsPerInt4 * 2];

        // 展开循环，处理数据打包
        {
            // 将int4值（128bit）转换为 bfloat162
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++k) { // 8
                // 将 bfloat162 转换为 float2
                const auto& fp322_fvalue = __bfloat1622float2(bf162_values[k]);

                /*
                实际进行压缩的公式为：
                q = clamp( round( (log2(abs(x)) - log_min) / (log_max - log_min) * (K - 2) + 0.5 ), 0, K - 1)
                其中：
                    x: 输入的浮点数
                    q: 输出的整数，表示压缩后的值
                    log_min: 输入中最小值的log2值
                    log_max: 输入中最大值的log2值
                    K: 压缩后的整数的最大值（即，K为2的幂）
                */
                // 对 float 值进行编码
                encoded[k * 2 + 0] = __float2uint_rd(__builtin_fmaxf(__builtin_amdgcn_logf(fp322_fvalue.x) * step_inv + fused_rounding, 0));
                encoded[k * 2 + 1] = __float2uint_rd(__builtin_fmaxf(__builtin_amdgcn_logf(fp322_fvalue.y) * step_inv + fused_rounding, 0));
            }

            // 批量打包编码后的值到 st_buffer
            st_buffer[0] = (encoded[0] >> 0) | (encoded[1] << 9) | (encoded[2] << 18) | (encoded[3] << 27);
            st_buffer[1] = (encoded[3] >> 5) | (encoded[4] << 4) | (encoded[5] << 13) | (encoded[6] << 22) | (encoded[7] << 31);
            st_buffer[2] = (encoded[7] >> 1) | (encoded[8] << 8) | (encoded[9] << 17) | (encoded[10] << 26);
            st_buffer[3] = (encoded[10] >> 6) | (encoded[11] << 3) | (encoded[12] << 12) | (encoded[13] << 21) | (encoded[14] << 30);
            st_buffer[4] = (encoded[14] >> 2) | (encoded[15] << 7) | (local_signs << 16);
        }
    }

    // 计算量化成功和失败时的数据量
    constexpr int unable_cast_num_bytes = kWarpSize * kSendValueBytes; // = 64*2*16 = 2048
    constexpr int enable_cast_num_bytes = unable_cast_num_bytes * 10 / 16; // = 2048/16*10=1280

    // Return TMA copy bytes
    return enable_cast ? enable_cast_num_bytes : unable_cast_num_bytes;
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_check_amaxmin(
    const uint8_t* meta_buffer, int4* shared_log_amax, int4* shared_log_amin, int* shared_cast_info, const int lane_id) {

    // 定义log阈值和最小剪切值
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumQuantGroupsPerWarp = kWarpSize / 16;
    using log_vec_type = int4;
    EP_STATIC_ASSERT(sizeof(log_vec_type) / sizeof(__hip_bfloat162) == kNumQuantGroupsPerWarp, "kNumQuantGroupsPerWarp == sizeof(log_vec_type) only");

    // 初始化类型转换启用标志
    bool enable_cast = true;

    // 如果 lane_id 小于 kNumLanes，则进行计算
    if (lane_id < kNumLanes) {
        // 从 meta_buffer 中读取 amaxmin2 值
        auto amaxmin4 = reinterpret_cast<const log_vec_type*>(meta_buffer)[lane_id];
        const auto& bf162_amaxmin = reinterpret_cast<__hip_bfloat162*>(&amaxmin4);

        // 定义 log_amax 和 log_amin 数组
        float log_amax[kNumQuantGroupsPerWarp], log_amin[kNumQuantGroupsPerWarp];

        // 展开循环，计算 log_amax 和 log_amin
        #pragma unroll
        for (int i = 0; i < kNumQuantGroupsPerWarp; ++i) {  // sizeof(uint64_t) / sizeof(__hip_bfloat162) = 2
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            log_amax[i] = __builtin_amdgcn_logf(amax);
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : __builtin_fmaxf(__builtin_amdgcn_logf(amin), log_amax[i] - kMinClip);

            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
        }

        // 将计算结果存储到 shared_log_amax 和 shared_log_amin 中
        int4 log_amax_int4 = *reinterpret_cast<int4*>(log_amax);
        int4 log_amin_int4 = *reinterpret_cast<int4*>(log_amin);
        shared_log_amax[lane_id] = log_amax_int4;
        shared_log_amin[lane_id] = log_amin_int4;
    }

    // 计算 casted 值。根据当前线程是否启用了类型转换，计算它所属的组的索引
    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls) : 0u;

    // 计算 num_casted_prefix 值。计算当前线程之前有多少个线程启用了类型转换。
    const auto& num_casted_prefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));

    // 如果 lane_id 小于 kNumLanes 且 lane_id 是 kNumRecvUnrolls 的倍数，则更新 shared_cast_info
    if (lane_id < kNumLanes and lane_id % kNumRecvUnrolls == 0) {
        // 最低1位保存casted结果，最高31位保存num_casted_prefix值
        shared_cast_info[lane_id / kNumRecvUnrolls] = (num_casted_prefix << 1) | (casted ? 1u : 0u);
    }
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate(
    const uint32_t* ld_buffer, float* accum, const float& log_amax, const float& log_amin, 
    const bool& enable_cast, const float& weight) {
    EP_STATIC_ASSERT(kNumRecvUnrolls == 2, "kNumRecvUnrolls == 2 only");

    if (enable_cast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto& step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
            const auto decoded = encoded == 0 ? .0f : __builtin_amdgcn_exp2f((encoded - 1) * step + log_amin);
            return sign ? -decoded : decoded;
        };

        uint32_t concat[6];
        concat[0] = ld_buffer[0];
        #pragma unroll
        for (int k = 1; k < 5; ++k)
            concat[k] = (ld_buffer[k - 1] >> (32 - k * 5)) | (ld_buffer[k] << (k * 5));
        concat[5] = ld_buffer[4] >> 7;

        const uint32_t& local_signs = ld_buffer[4] >> 16;
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            accum[k * 3 + 0] += decode((concat[k] >> 0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
            accum[k * 3 + 1] += decode((concat[k] >> 9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
            accum[k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
        }
        accum[15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
    } else {
        constexpr int kLoopIter = kNumRecvUnrolls * sizeof(int4) / sizeof(uint32_t);
        #pragma unroll
        for (int k = 0; k < kLoopIter; ++k) {
            auto bf16_pack = *reinterpret_cast<const __hip_bfloat162*>(ld_buffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
}

} // namespace internode_ll

} // namespace deep_ep
