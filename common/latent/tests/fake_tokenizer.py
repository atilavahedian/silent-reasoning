"""A tiny deterministic tokenizer for CPU tests. Satisfies the protocol in common/latent/INTERFACE.md.

Whitespace tokens with a growing vocabulary; punctuation is split off so "### 42" and "lempus." behave like words.
`add_special_tokens` is accepted and ignored (GPT-2 adds no BOS either). Special tokens are appended with add_tokens.
"""
import re

_SPLIT = re.compile(r"\s+|(?=[.,?#])|(?<=[.,?#])")


class FakeTokenizer:
    def __init__(self):
        self.id_to_tok = ["<eos>", "<pad>"]
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.padding_side = "right"

    def _pieces(self, text):
        return [p for p in _SPLIT.split(text) if p and not p.isspace()]

    def add_tokens(self, toks):
        n = 0
        for t in toks:
            if t not in self.tok_to_id:
                self.tok_to_id[t] = len(self.id_to_tok)
                self.id_to_tok.append(t)
                n += 1
        return n

    def convert_tokens_to_ids(self, tok):
        return self.tok_to_id[tok]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for p in self._pieces(text):
            if p not in self.tok_to_id:
                self.tok_to_id[p] = len(self.id_to_tok)
                self.id_to_tok.append(p)
            ids.append(self.tok_to_id[p])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.id_to_tok[i] for i in ids)

    def __len__(self):
        return len(self.id_to_tok)
