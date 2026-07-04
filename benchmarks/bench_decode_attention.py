"""Micro-benchmark: CUDA flash-decode kernel, heuristic vs fixed-256 splits.

Shows the performance rationale for LLM-42: the batch-adaptive split heuristic
(nondeterministic) vs the batch-invariant fixed tiling, across batch sizes and
context lengths.

    python benchmarks/bench_decode_attention.py
"""

from __future__ import annotations

import time

import torch

HQ, HKV, D = 16, 8, 128
SCALE = D ** -0.5


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    from sid.kernels.decode_attention import DecodeAttention

    attn = DecodeAttention(max_batch=256, num_q_heads=HQ, num_kv_heads=HKV,
                           head_dim=D, max_kv_splits=32, invariant_split_size=256,
                           max_seq_len=32768, device="cuda")

    print(f"{'batch':>6} {'seq':>7} {'heuristic us':>13} {'fixed256 us':>12} {'sdpa us':>9}")
    for bs in (1, 4, 16, 64):
        for seq in (128, 1024, 8192):
            torch.manual_seed(0)
            total = bs * seq
            q = torch.randn(bs, HQ, D, device="cuda", dtype=torch.bfloat16)
            k_cache = torch.randn(total + 8, HKV, D, device="cuda", dtype=torch.bfloat16)
            v_cache = torch.randn(total + 8, HKV, D, device="cuda", dtype=torch.bfloat16)
            kv_indices = torch.arange(1, total + 1, device="cuda", dtype=torch.int32)
            kv_indptr = torch.arange(0, total + 1, seq, device="cuda", dtype=torch.int32)
            seq_lens = torch.full((bs,), seq, device="cuda", dtype=torch.int32)

            t_heur = bench(lambda: attn.forward(
                q, k_cache, v_cache, kv_indptr, kv_indices, seq_lens, seq,
                SCALE, invariant=False))
            t_fix = bench(lambda: attn.forward(
                q, k_cache, v_cache, kv_indptr, kv_indices, seq_lens, seq,
                SCALE, invariant=True))

            # SDPA reference (dense layout, batch-strided) as a speed baseline.
            kd = k_cache[1:total + 1].view(bs, seq, HKV, D).permute(0, 2, 1, 3)
            vd = v_cache[1:total + 1].view(bs, seq, HKV, D).permute(0, 2, 1, 3)
            qd = q.view(bs, 1, HQ, D).permute(0, 2, 1, 3)
            t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
                qd, kd, vd, enable_gqa=True))

            print(f"{bs:>6} {seq:>7} {t_heur:>13.1f} {t_fix:>12.1f} {t_sdpa:>9.1f}")


if __name__ == "__main__":
    main()
