"""ModelRunner: owns the model, KV cache, attention workspaces, and the
numerics environment. One forward = one ForwardBatch.

Numerics pinning (all modes): TF32 off, reduced-precision bf16 reductions off,
CUBLAS_WORKSPACE_CONFIG pinned before the first cuBLAS call. On SM90 this
makes cuBLAS GEMMs run-consistent for a fixed shape — which is exactly the
property the DVR verifier's fixed G x W shape converts into determinism.
"""

from __future__ import annotations

import logging
import os

import torch

from sid.config import EngineConfig, Mode, ModelConfig
from sid.engine.batch import ForwardBatch
from sid.engine.kv_cache import KVCache
from sid.engine.sampler import sample as _sample
from sid.kernels.decode_attention import DecodeAttention
from sid.model.loader import load_model_config, load_weights, resolve_model_dir
from sid.model.qwen3 import Qwen3Model

logger = logging.getLogger(__name__)


def pin_numerics() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


class ModelRunner:
    def __init__(self, cfg: EngineConfig, device: str = "cuda"):
        pin_numerics()
        self.cfg = cfg
        self.device = device

        model_dir = resolve_model_dir(cfg.model_path)
        self.model_dir = model_dir
        self.mcfg: ModelConfig = load_model_config(model_dir, cfg.model_path)
        logger.info("loading %s (%d layers, %d/%d heads, head_dim %d)",
                    cfg.model_path, self.mcfg.num_layers,
                    self.mcfg.num_q_heads, self.mcfg.num_kv_heads, self.mcfg.head_dim)

        weights = load_weights(model_dir, self.mcfg, device)
        self.model = Qwen3Model(self.mcfg, weights, device)
        self.model.set_batch_invariant(cfg.mode == Mode.BATCH_INVARIANT)

        self.kv_cache = KVCache(cfg, self.mcfg, device)
        self.decode_attn = DecodeAttention(
            max_batch=cfg.max_decode_batch,
            num_q_heads=self.mcfg.num_q_heads,
            num_kv_heads=self.mcfg.num_kv_heads,
            head_dim=self.mcfg.head_dim,
            max_kv_splits=cfg.max_kv_splits,
            invariant_split_size=cfg.invariant_split_size,
            max_seq_len=cfg.max_seq_len,
            device=device,
        )

        # Persistent verify-pass input buffers (fixed base pointers so verify
        # passes see identical tensor addresses/alignment every time).
        gw = cfg.dvr_group_size * cfg.dvr_window_size
        self.verify_buffers = {
            "input_ids": torch.zeros(gw, dtype=torch.int64, device=device),
            "positions": torch.zeros(gw, dtype=torch.int64, device=device),
            "out_slots": torch.zeros(gw, dtype=torch.int64, device=device),
            "sample_positions": torch.zeros(gw, dtype=torch.int64, device=device),
            "temperatures": torch.zeros(gw, dtype=torch.float32, device=device),
            "top_ks": torch.zeros(gw, dtype=torch.int64, device=device),
            "top_ps": torch.zeros(gw, dtype=torch.float32, device=device),
            "seeds": torch.zeros(gw, dtype=torch.int64, device=device),
            "qo_indptr": torch.zeros(cfg.dvr_group_size + 1, dtype=torch.int32, device=device),
            "prefix_kv_indptr": torch.zeros(cfg.dvr_group_size + 1, dtype=torch.int32, device=device),
            "prefix_kv_indices": torch.zeros(cfg.dvr_group_size * cfg.max_seq_len,
                                             dtype=torch.int32, device=device),
        }

    def forward(self, fb: ForwardBatch) -> torch.Tensor:
        return self.model.forward(fb, self.kv_cache, self.decode_attn,
                                  decode_backend=self.cfg.decode_backend)

    def sample(self, fb: ForwardBatch, logits: torch.Tensor) -> torch.Tensor:
        return _sample(logits, fb.temperatures, fb.top_ks, fb.top_ps,
                       fb.seeds, fb.sample_positions)
