"""Minimal whitespace/alphanumeric tokenizer + vocabulary, built from the
training dataset's own prompts.

No pretrained tokenizer or embeddings -- a pretrained language model would
be complete overkill for (and wouldn't meaningfully help with) a training
set of tens-to-hundreds of samples. This is a standard small-data baseline
(bag-of-embeddings over a dataset-specific vocabulary), not a corner cut."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")

PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Vocabulary:
    token_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1) -> "Vocabulary":
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        tokens = [PAD, UNK] + sorted(tok for tok, c in counts.items() if c >= min_freq)
        return cls(token_to_id={tok: i for i, tok in enumerate(tokens)})

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    def encode(self, text: str, max_len: int) -> list[int]:
        unk_id = self.token_to_id[UNK]
        ids = [self.token_to_id.get(tok, unk_id) for tok in tokenize(text)][:max_len]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def to_dict(self) -> dict:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        return cls(token_to_id=dict(d["token_to_id"]))
