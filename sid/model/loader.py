"""Model loading: HF config.json + safetensors -> ModelConfig + weight dict.

QKV and gate/up projections are fused at load time (fewer GEMM launches).
Qwen3-0.6B has tied embeddings: the lm_head weight IS embed_tokens.weight and
does not exist separately in the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sid.config import ModelConfig


def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model_path,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "merges.txt", "vocab*"],
    ))


def load_model_config(model_dir: Path, model_path: str) -> ModelConfig:
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "qwen3", f"only qwen3 supported, got {cfg['model_type']}"

    eos_ids = [cfg.get("eos_token_id", 151645)]
    gen_cfg_path = model_dir / "generation_config.json"
    if gen_cfg_path.exists():
        gen = json.loads(gen_cfg_path.read_text())
        eos = gen.get("eos_token_id", eos_ids)
        eos_ids = eos if isinstance(eos, list) else [eos]

    return ModelConfig(
        model_path=model_path,
        num_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        num_q_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],  # explicit — do NOT derive from hidden/heads
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        eos_token_ids=tuple(eos_ids),
    )


def load_weights(model_dir: Path, mcfg: ModelConfig, device: str,
                 dtype: torch.dtype = torch.bfloat16) -> dict:
    """Returns {embed, norm, lm_head, layers: [ {qkv, o, q_norm, k_norm,
    input_ln, post_ln, gate_up, down} ]} on the target device."""
    from safetensors.torch import load_file

    raw: dict[str, torch.Tensor] = {}
    files = sorted(model_dir.glob("*.safetensors"))
    assert files, f"no safetensors found in {model_dir}"
    for f in files:
        raw.update(load_file(str(f)))

    def take(name: str) -> torch.Tensor:
        return raw.pop(name).to(device=device, dtype=dtype)

    weights: dict = {"layers": []}
    weights["embed"] = take("model.embed_tokens.weight")
    weights["norm"] = take("model.norm.weight")
    if mcfg.tie_word_embeddings:
        weights["lm_head"] = weights["embed"]
    else:
        weights["lm_head"] = take("lm_head.weight")

    for i in range(mcfg.num_layers):
        p = f"model.layers.{i}."
        layer = {
            "qkv": torch.cat(
                [take(p + "self_attn.q_proj.weight"),
                 take(p + "self_attn.k_proj.weight"),
                 take(p + "self_attn.v_proj.weight")], dim=0
            ).contiguous(),
            "o": take(p + "self_attn.o_proj.weight"),
            "q_norm": take(p + "self_attn.q_norm.weight"),
            "k_norm": take(p + "self_attn.k_norm.weight"),
            "input_ln": take(p + "input_layernorm.weight"),
            "post_ln": take(p + "post_attention_layernorm.weight"),
            "gate_up": torch.cat(
                [take(p + "mlp.gate_proj.weight"),
                 take(p + "mlp.up_proj.weight")], dim=0
            ).contiguous(),
            "down": take(p + "mlp.down_proj.weight"),
        }
        weights["layers"].append(layer)

    leftovers = [k for k in raw if "rotary" not in k]
    assert not leftovers, f"unconsumed checkpoint tensors: {leftovers[:5]}"
    return weights
