import argparse
import random
import os
import torch
import torch.distributed as dist
from functools import partial
from typing import Literal, Set

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back_int8


def test_main(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    # # NOTES: the last one is for performance testing
    # # Most of the values in the perf case is lower than the threshold, casting most channels
    # x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for _ in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    # For failure simulation and shrink testing
    mask_status = torch.zeros((num_ranks,), dtype=torch.int, device='cuda')

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for quant_type in (1, ):
                for fp8_round_scale in (False, ):
                    for quant_group_size in (0, ):
                        dispatch_use_fp8 = quant_type > 0
                        num_times += 1
                        for _ in range(1):
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            quant_type=quant_type, quant_group_size=quant_group_size,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()

                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        simulated_gemm_x = per_token_cast_back_int8(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, 1)).view(packed_recv_x[0].shape)

                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i

                            recv_x = per_token_cast_back_int8(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # Check expert indices
                            int_mask = (2 ** 32) - 1
                            num_valid_tokens = recv_count.item()
                            assert num_valid_tokens == (
                                    recv_layout_range
                                    & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item(
                            ), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item()}'

                            if num_valid_tokens == 0:
                                continue

                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_x_amax = recv_x[:, :-128].amax(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]

                                assert torch.equal(recv_x_amin, recv_x_amax)

                                if quant_type == 1:
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.01
                                else:
                                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0

                                # for j in range(num_ranks):
                                #     if (not round_scale):
                                #         check_tmp1 = (recv_x_amin == j - rank_offset).sum().item()
                                #         check_tmp2 = (all_topk_idx[j] == expert_id).sum().item()
                                #         print(f'rank: {rank}, j: {j}, check_tmp1: {check_tmp1}, check_tmp2: {check_tmp2}, diff: {abs(check_tmp1 - check_tmp2)}')
                                #         assert abs(check_tmp1 - check_tmp2) < 3
                                #         assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0

                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                            else:
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

    print("dispatch int 8 pass")

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        quant_type=1, quant_group_size=0,
                                        async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                             topk_idx,
                                                             topk_weights,
                                                             handle,
                                                             return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    scale_size = 1  # hidden / 128
    num_fp8_bytes, num_bf16_bytes = (hidden + scale_size * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Separate profiling
    for return_recv_hook in (True, False):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'),
                                             barrier_comm_profiling=True,
                                             suppress_kineto_output=True,
                                             num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us',
                  flush=True)
        else:
            print(f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                  f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us',
                  flush=True)
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')
    parser.add_argument("--pressure-test", action='store_true', help='Whether to do pressure test')
    parser.add_argument("--shrink-test", action='store_true', help='Whether to simulate failure and test shrink mode')
    parser.add_argument('--use-logfmt', action='store_true', help='Whether to test LogFMT combine')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
