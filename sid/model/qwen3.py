"""Qwen3 dense model forward over a ForwardBatch.

Qwen3 specifics handled here (all verified against sglang/models/qwen3.py and
HF modeling_qwen3):
  - head_dim (128) is explicit in the config, NOT hidden_size // num_heads
  - per-head QK-RMSNorm applied after the QKV projection, BEFORE RoPE
  - no QKV bias; SwiGLU MLP; pre-norm with fused residual-add RMSNorm
  - tied lm_head (embed_tokens.weight)

Attention routing:
  DECODE          -> CUDA split-KV flash-decode kernel (or Triton extend as a
                     debugging fallback via decode_backend="triton")
  PREFILL/VERIFY  -> Triton extend attention (fixed tiles)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sid.config import ModelConfig
from sid.engine.batch import ForwardBatch, ForwardMode
from sid.kernels.extend_attention import extend_attention_fwd
from sid.kernels.matmul_persistent import bi_linear
from sid.kernels.rmsnorm import fused_add_rmsnorm, qk_head_rmsnorm, rmsnorm
from sid.kernels.rope import RotaryEmbedding


class Qwen3Model:
    def __init__(self, mcfg: ModelConfig, weights: dict, device: str):
        self.cfg = mcfg
        self.w = weights
        self.device = device
        self.rope = RotaryEmbedding(
            mcfg.head_dim, mcfg.max_position_embeddings, mcfg.rope_theta, device
        )
        self.sm_scale = mcfg.head_dim ** -0.5
        self.linear = F.linear  # swapped to bi_linear in batch_invariant mode

    def set_batch_invariant(self, enabled: bool) -> None:
        self.linear = bi_linear if enabled else F.linear

    def _attention(self, layer_idx: int, hidden: torch.Tensor, fb: ForwardBatch,
                   kv_cache, decode_attn, decode_backend: str) -> torch.Tensor:
        cfg = self.cfg
        lw = self.w["layers"][layer_idx]
        t = hidden.shape[0]

        qkv = self.linear(hidden, lw["qkv"])
        q, k, v = qkv.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)
        q = q.contiguous().view(t, cfg.num_q_heads, cfg.head_dim)
        k = k.contiguous().view(t, cfg.num_kv_heads, cfg.head_dim)
        v = v.contiguous().view(t, cfg.num_kv_heads, cfg.head_dim)

        # QK-norm BEFORE RoPE (Qwen3).
        q = qk_head_rmsnorm(q, lw["q_norm"], cfg.rms_norm_eps)
        k = qk_head_rmsnorm(k, lw["k_norm"], cfg.rms_norm_eps)
        self.rope.apply(q, k, fb.positions)

        k_pool, v_pool = kv_cache.layer(layer_idx)
        kv_cache.ext.reshape_and_cache(k, v, k_pool, v_pool, fb.out_slots)

        if fb.mode == ForwardMode.DECODE and decode_backend == "cuda":
            o = decode_attn.forward(
                q, k_pool, v_pool,
                fb.kv_indptr, fb.kv_indices,
                fb.seq_lens, fb.max_seq_len,
                self.sm_scale, invariant=fb.invariant,
            )
        else:
            # PREFILL / VERIFY (and the Triton decode fallback): extend
            # attention over the paged prefix + fresh extend tokens.
            o = extend_attention_fwd(
                q, k, v, k_pool, v_pool,
                fb.qo_indptr, fb.prefix_kv_indptr, fb.prefix_kv_indices,
                fb.max_extend_len, self.sm_scale,
            )

        return self.linear(o.view(t, cfg.q_size), lw["o"])

    def _mlp(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        lw = self.w["layers"][layer_idx]
        gate_up = self.linear(hidden, lw["gate_up"])
        gate, up = gate_up.chunk(2, dim=-1)
        return self.linear(F.silu(gate) * up, lw["down"])

    @torch.inference_mode()
    def forward(self, fb: ForwardBatch, kv_cache, decode_attn,
                decode_backend: str = "cuda") -> torch.Tensor:
        """Returns fp32 logits [len(fb.sample_indices), vocab]."""
        cfg = self.cfg
        hidden = F.embedding(fb.input_ids, self.w["embed"])
        residual = None

        for i in range(cfg.num_layers):
            lw = self.w["layers"][i]
            if residual is None:
                residual = hidden
                hidden = rmsnorm(hidden, lw["input_ln"], cfg.rms_norm_eps)
            else:
                hidden, residual = fused_add_rmsnorm(
                    hidden, residual, lw["input_ln"], cfg.rms_norm_eps
                )
            hidden = self._attention(i, hidden, fb, kv_cache, decode_attn, decode_backend)
            hidden, residual = fused_add_rmsnorm(
                hidden, residual, lw["post_ln"], cfg.rms_norm_eps
            )
            hidden = self._mlp(i, hidden)

        hidden, _ = fused_add_rmsnorm(hidden, residual, self.w["norm"], cfg.rms_norm_eps)

        rows = hidden[fb.sample_indices].contiguous()
        logits = self.linear(rows, self.w["lm_head"])
        return logits.float()
