"""Demo prompt set: one watched prompt + fillers, and the batch compositions.

Compositions deliberately vary batch SIZE, not just order: most kernels are
position-invariant (LLM-42 paper, Observation O2), so permuting an identical
prompt set often agrees even in nondeterministic mode. What changes results is
different co-scheduled batch sizes/compositions — which change the decode
kernel's split-KV count and the GEMM shapes over the course of generation.
"""

WATCHED = (
    "Give a concise two-sentence explanation of why deterministic LLM "
    "inference matters for production systems."
)

FILLERS = [
    "Write a short haiku about a database migration finishing cleanly.",
    "List three practical checks before deploying a model-serving change.",
    "Explain request batching to a new ML infrastructure engineer.",
    "Summarize the idea of speculative decoding in three sentences.",
    "Describe the water cycle in exactly four sentences.",
    "Explain what a page table does in an operating system.",
    "Write two sentences about why GPUs are good at matrix multiplication. "
    * 8,  # long filler: stretches prefill shapes
]

PROMPTS = [WATCHED] + FILLERS

# Index lists into PROMPTS; 0 is the watched prompt.
COMPOSITIONS = [
    ("alone",          [0]),
    ("batch4-a",       [0, 1, 2, 3]),
    ("batch4-b",       [3, 0, 1, 2]),      # same set, different order
    ("batch4-c",       [2, 1, 4, 0]),      # different member set
    ("batch6",         [0, 1, 2, 3, 4, 5]),
    ("batch16-mixed",  [0] + [1, 2, 3, 4, 5, 6, 7] * 2 + [6]),
]
