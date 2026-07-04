"""Incremental detokenization gated at the DVR verification frontier.

The visible token prefix is monotone (verified tokens are never rolled back),
so re-decoding the visible prefix and emitting the string delta is correct.
We hold back the last few characters when the tokenizer may still merge them
(the standard trick for multi-byte/merged pieces is unnecessary here because
we always re-decode the full visible prefix — a later call can only extend it).
"""

from __future__ import annotations


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._emitted = ""

    def update(self, visible_ids: list) -> str:
        """Returns the newly visible text (may be empty)."""
        text = self.tokenizer.decode(visible_ids, skip_special_tokens=True)
        # Don't emit a trailing replacement char from a half-decoded byte pair.
        while text.endswith("�"):
            text = text[:-1]
        if not text.startswith(self._emitted):
            # Extremely rare (visible prefix is monotone); resync conservatively.
            delta = text
        else:
            delta = text[len(self._emitted):]
        self._emitted = text if len(text) >= len(self._emitted) else self._emitted
        return delta
