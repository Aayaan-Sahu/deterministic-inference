"""Configuration for the sid engine.

Three execution modes (see README):
  - nondet:          fast path only. Decode attention picks its split-KV count from
                     batch size / SM occupancy (like FA3 / sglang), so outputs depend
                     on batch composition. Fast, NOT deterministic.
  - dvr:             LLM-42 decode-verify-rollback. Fast path identical to nondet,
                     plus fixed-shape verification passes that enforce determinism
                     for requests with is_deterministic=True.
  - batch_invariant: every kernel uses a batch-independent reduction order
                     (fixed split tiles, persistent Triton GEMM). Deterministic by
                     construction, slower.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class Mode(str, enum.Enum):
    NONDET = "nondet"
    DVR = "dvr"
    BATCH_INVARIANT = "batch_invariant"


# Token used to pad verification windows and dummy verification requests.
# Same value as the LLM-42 reference implementation.
DUMMY_TOKEN_ID = 32


@dataclass
class ModelConfig:
    """Filled from the HF config.json by sid.model.loader."""

    model_path: str
    num_layers: int = 28
    hidden_size: int = 1024
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128  # explicit in Qwen3 config; NOT hidden_size // num_q_heads
    intermediate_size: int = 3072
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    eos_token_ids: tuple = (151645, 151643)

    @property
    def q_size(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass
class EngineConfig:
    model_path: str = "Qwen/Qwen3-0.6B"
    mode: Mode = Mode.DVR
    dtype: str = "bfloat16"

    # KV cache (256K tokens x ~114KB/token for Qwen3-0.6B ≈ 29 GB on the GPU)
    kv_pool_tokens: int = 256 * 1024  # slots in the KV pool (slot 0 reserved)
    max_reqs: int = 256
    max_seq_len: int = 32768  # req_to_token row width
    max_prompt_len: int = 16384  # longer prompts are rejected (no chunked prefill)

    # scheduling
    max_prefill_tokens: int = 8192  # token budget for a batched (non-isolated) prefill
    max_decode_batch: int = 256

    # DVR
    dvr_window_size: int = 32  # W: tokens verified per request per pass
    dvr_group_size: int = 4  # G: requests per fixed-shape verification pass

    # decode attention split-KV
    max_kv_splits: int = 32  # heuristic mode upper bound
    invariant_split_size: int = 256  # fixed tile for batch_invariant mode

    # debugging
    decode_backend: str = "cuda"  # "cuda" | "triton" (fallback for bisecting kernel bugs)

    device: str = "cuda"


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 => greedy argmax
    top_k: int = -1  # -1 => disabled
    top_p: float = 1.0
    max_new_tokens: int = 128
    seed: int = 42
    is_deterministic: bool = False
    stop_token_ids: Optional[list] = None
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
