"""Phase 8: DVR bookkeeping on CPU with fake KV/runner — no GPU needed.

Exercises the window-construction and rollback arithmetic that the paper's
correctness rests on: slot reuse (overwrite!), padding, the P+N-1 KV
invariant, EOS-clear/re-finish, and verified_len monotonicity.
"""

from __future__ import annotations

import torch

from sid.config import DUMMY_TOKEN_ID, EngineConfig, Mode, SamplingParams
from sid.engine.dvr import Verifier
from sid.engine.request import Request

W, G = 8, 2
EOS = 9


class FakeKV:
    def __init__(self, rows=32, width=256, slots=10000):
        self.req_to_token = torch.zeros(rows, width, dtype=torch.int32)
        self.device = "cpu"
        self._next = 1
        self.freed: list[int] = []

    def alloc(self, n: int) -> torch.Tensor:
        out = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return out

    def free(self, slots: torch.Tensor) -> None:
        self.freed.extend(slots.tolist())


class FakeRunner:
    def __init__(self, cfg: EngineConfig):
        gw = cfg.dvr_group_size * cfg.dvr_window_size
        g = cfg.dvr_group_size
        self.verify_buffers = {
            "input_ids": torch.zeros(gw, dtype=torch.int64),
            "positions": torch.zeros(gw, dtype=torch.int64),
            "out_slots": torch.zeros(gw, dtype=torch.int64),
            "sample_positions": torch.zeros(gw, dtype=torch.int64),
            "temperatures": torch.zeros(gw, dtype=torch.float32),
            "top_ks": torch.zeros(gw, dtype=torch.int64),
            "top_ps": torch.zeros(gw, dtype=torch.float32),
            "seeds": torch.zeros(gw, dtype=torch.int64),
            "qo_indptr": torch.zeros(g + 1, dtype=torch.int32),
            "prefix_kv_indptr": torch.zeros(g + 1, dtype=torch.int32),
            "prefix_kv_indices": torch.zeros(g * 256, dtype=torch.int32),
        }


def make_setup():
    cfg = EngineConfig(mode=Mode.DVR, dvr_window_size=W, dvr_group_size=G,
                       device="cpu")
    kv = FakeKV()
    runner = FakeRunner(cfg)
    verifier = Verifier(cfg, runner, kv)
    return cfg, kv, runner, verifier


def make_request(kv: FakeKV, rid: int, prompt_len: int, outputs: list[int],
                 verified: int, row: int) -> Request:
    """Request in post-decode state: KV valid for positions 0..P+N-2."""
    req = Request(rid, list(range(100, 100 + prompt_len)),
                  SamplingParams(is_deterministic=True, max_new_tokens=1000),
                  eos_token_ids=(EOS,))
    req.output_ids = list(outputs)
    req.verified_len = verified
    req.req_row = row
    kv_len = prompt_len + len(outputs) - 1
    kv.req_to_token[row, :kv_len] = torch.arange(1000, 1000 + kv_len, dtype=torch.int32)
    req.check_finished()
    return req


# ---------------------------------------------------------------------------


def test_window_construction_full_window():
    cfg, kv, runner, verifier = make_setup()
    p, v = 5, 1
    outputs = list(range(50, 59))  # 9 outputs, unverified = 8 = W
    req = make_request(kv, 0, p, outputs, v, row=3)
    assert req.unverified() == W

    fb, meta = verifier.build_verify_batch([req])
    buf = runner.verify_buffers

    # input row 0: [last verified token] + window[:-1]
    assert buf["input_ids"][:W].tolist() == [outputs[0]] + outputs[1:8]
    # positions P+v-1 .. P+v+W-2
    assert buf["positions"][:W].tolist() == list(range(p + v - 1, p + v - 1 + W))
    # sample positions are one ahead (position of the token being generated)
    assert buf["sample_positions"][:W].tolist() == list(range(p + v, p + v + W))
    # out_slots REUSE decode's physical slots (the overwrite trick)
    expected_slots = kv.req_to_token[3, p + v - 1:p + v - 1 + W].tolist()
    assert buf["out_slots"][:W].tolist() == expected_slots
    # prefix covers positions 0..P+v-2
    assert buf["prefix_kv_indptr"].tolist()[:2] == [0, p + v - 1]
    assert buf["prefix_kv_indices"][:p + v - 1].tolist() == \
        kv.req_to_token[3, :p + v - 1].tolist()
    # dummy sequence fills the second slot of the group
    assert buf["input_ids"][W:2 * W].tolist() == [DUMMY_TOKEN_ID] * W
    assert meta.window_lens == [W]
    assert meta.windows == [outputs[1:9]]


def test_window_construction_finished_with_padding():
    cfg, kv, runner, verifier = make_setup()
    p = 5
    outputs = [50, 51, 52, 53, 54, EOS]  # finished, 6 outputs
    req = make_request(kv, 1, p, outputs, verified=3, row=4)
    assert req.finished_reason == "stop"
    u = req.unverified()
    assert u == 3

    fb, meta = verifier.build_verify_batch([req])
    buf = runner.verify_buffers

    # [context=out[2]] + window[:-1]=out[3:5] + DUMMY padding
    assert buf["input_ids"][:W].tolist() == \
        [outputs[2]] + outputs[3:5] + [DUMMY_TOKEN_ID] * (W - u)
    # real slots only for the u available positions, then scratch padding
    kv_len = p + len(outputs) - 1
    real = kv.req_to_token[4, p + 3 - 1:p + 3 - 1 + u].tolist()
    got = buf["out_slots"][:W].tolist()
    assert got[:u] == real
    assert all(s not in kv.req_to_token[4, :kv_len].tolist() for s in got[u:]), \
        "padding positions must NOT touch the request's real KV slots"


def test_all_match_advances_verified_len():
    cfg, kv, runner, verifier = make_setup()
    outputs = list(range(50, 59))
    req = make_request(kv, 2, 5, outputs, 1, row=5)
    fb, meta = verifier.build_verify_batch([req])

    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[:W] = torch.tensor(outputs[1:9])
    verifier.compare_and_rollback(meta, tokens)

    assert req.verified_len == 9
    assert req.output_ids == outputs
    assert req.num_rollbacks == 0
    assert kv.freed == []


def test_mismatch_middle_rolls_back_and_accepts_verifier_token():
    cfg, kv, runner, verifier = make_setup()
    p, v = 5, 1
    outputs = list(range(50, 59))  # window = outputs[1:9]
    req = make_request(kv, 3, p, outputs, v, row=6)
    fb, meta = verifier.build_verify_batch([req])

    verify_out = outputs[1:9].copy()
    verify_out[3] = 777  # mismatch at window index 3
    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[:W] = torch.tensor(verify_out)
    old_len = len(outputs)
    verifier.compare_and_rollback(meta, tokens)

    # keep v+j = 4 tokens, then the verifier's token
    assert req.output_ids == outputs[:4] + [777]
    assert req.verified_len == 5
    assert req.num_rollbacks == 1
    assert req.tokens_rolled_back == old_len - (v + 3)
    # freed KV: positions P+new_len-1 .. P+old_len-2 = 9..12
    expected_freed = kv.req_to_token[6, 9:13].tolist()
    assert kv.freed == expected_freed
    # invariant: KV frontier matches the new output length
    assert req.kv_len == p + len(req.output_ids) - 1
    assert req.finished_reason is None


def test_mismatch_at_zero_still_progresses():
    cfg, kv, runner, verifier = make_setup()
    outputs = list(range(50, 59))
    req = make_request(kv, 4, 5, outputs, 1, row=7)
    fb, meta = verifier.build_verify_batch([req])

    verify_out = [666] + outputs[2:9]
    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[:W] = torch.tensor(verify_out)
    verifier.compare_and_rollback(meta, tokens)

    assert req.output_ids == outputs[:1] + [666]  # >= 1 token progress
    assert req.verified_len == 2


def test_mismatch_last_token_frees_nothing():
    cfg, kv, runner, verifier = make_setup()
    p, v = 5, 1
    outputs = list(range(50, 59))
    req = make_request(kv, 5, p, outputs, v, row=8)
    fb, meta = verifier.build_verify_batch([req])

    verify_out = outputs[1:9].copy()
    verify_out[-1] = 888  # j = u-1
    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[:W] = torch.tensor(verify_out)
    verifier.compare_and_rollback(meta, tokens)

    assert req.output_ids == outputs[:8] + [888]
    assert req.verified_len == 9
    assert kv.freed == []  # new frontier == old frontier


def test_eos_cleared_then_refinished_on_corrected_eos():
    cfg, kv, runner, verifier = make_setup()
    p = 5
    outputs = [50, 51, 52, 53, 54, EOS]
    req = make_request(kv, 6, p, outputs, verified=3, row=9)
    assert req.finished_reason == "stop"
    fb, meta = verifier.build_verify_batch([req])

    # Verifier disagrees at index 0 of the window and emits EOS there.
    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[0] = EOS
    tokens[1:3] = torch.tensor([54, EOS])
    verifier.compare_and_rollback(meta, tokens)

    assert req.output_ids == outputs[:3] + [EOS]
    assert req.finished_reason == "stop"  # re-finished on the corrected EOS
    assert req.verified_len == 4
    assert req.unverified() == 0  # fully verified -> releasable


def test_eos_cleared_and_decoding_resumes():
    cfg, kv, runner, verifier = make_setup()
    p = 5
    outputs = [50, 51, 52, 53, 54, EOS]
    req = make_request(kv, 7, p, outputs, verified=3, row=10)
    fb, meta = verifier.build_verify_batch([req])

    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[0:3] = torch.tensor([53, 54, 42])  # disagrees on the EOS itself
    verifier.compare_and_rollback(meta, tokens)

    assert req.output_ids[-1] == 42
    assert req.finished_reason is None  # must resume decoding
    assert req.verified_len == len(req.output_ids)


def test_u_equals_one_boundary():
    cfg, kv, runner, verifier = make_setup()
    p = 5
    outputs = [50, EOS]
    req = make_request(kv, 8, p, outputs, verified=1, row=11)
    assert req.unverified() == 1
    fb, meta = verifier.build_verify_batch([req])
    buf = runner.verify_buffers
    # input = [context] + 0 window tokens + W-1 dummies
    assert buf["input_ids"][:W].tolist() == [50] + [DUMMY_TOKEN_ID] * (W - 1)

    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[0] = EOS
    verifier.compare_and_rollback(meta, tokens)
    assert req.verified_len == 2 and req.output_ids == outputs


def test_two_requests_grouped_offsets():
    cfg, kv, runner, verifier = make_setup()
    r1 = make_request(kv, 10, 4, list(range(50, 59)), 1, row=12)
    r2 = make_request(kv, 11, 7, list(range(70, 79)), 1, row=13)
    fb, meta = verifier.build_verify_batch([r1, r2])

    tokens = torch.zeros(G * W, dtype=torch.int64)
    tokens[:W] = torch.tensor(r1.output_ids[1:9])          # r1 all match
    bad = r2.output_ids[1:9].copy()
    bad[2] = 999                                            # r2 mismatch at 2
    tokens[W:2 * W] = torch.tensor(bad)
    verifier.compare_and_rollback(meta, tokens)

    assert r1.verified_len == 9 and r1.num_rollbacks == 0
    assert r2.output_ids == list(range(70, 73)) + [999]
    assert r2.verified_len == 4


def test_verified_len_monotone_over_many_windows():
    cfg, kv, runner, verifier = make_setup()
    req = make_request(kv, 12, 5, list(range(50, 59)), 1, row=14)
    history = [req.verified_len]
    for trial in range(3):
        if req.unverified() < 1:
            break
        fb, meta = verifier.build_verify_batch([req])
        u = meta.window_lens[0]
        verify_out = meta.windows[0].copy()
        if trial == 0 and u > 2:
            verify_out[2] = 300 + trial
        tokens = torch.zeros(G * W, dtype=torch.int64)
        tokens[:u] = torch.tensor(verify_out[:u])
        verifier.compare_and_rollback(meta, tokens)
        history.append(req.verified_len)
        # regrow outputs to simulate more decoding
        while req.unverified() < W and len(req.output_ids) < 20:
            req.output_ids.append(60 + len(req.output_ids))
            pos = req.prompt_len + len(req.output_ids) - 2
            kv.req_to_token[14, pos] = 2000 + pos
    assert history == sorted(history), f"verified_len not monotone: {history}"
