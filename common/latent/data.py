"""Coconut-format data: loading, tokenization, per-stage datasets for every arm, and the batch collator.

Mirrors facebookresearch/coconut (MIT) `dataset.py` (get_dataset, get_cot_latent_dataset, get_question_latent_dataset,
MyCollator) and the stage/arm selection and answer extraction of `run.py`, with the HF `datasets` dependency replaced
by plain lists. Token-for-token equality with the reference is asserted in `tests/test_data.py` against the port in
`tests/reference_dataset.py`.

Record layout produced by `build_stage_dataset` (training records, for_generation=False):

    input_ids = question + [<start>] + n_latent * [<latent>] + [<end>] + steps[n_skip:] + answer
    labels    = -100 * (len(question) + n_latent + n_markers) + input_ids[len(question) + n_latent + n_markers:]

where `question` is `encode(question + "\n", add_special_tokens=True)`, each step is `encode(step + "\n")`, `answer` is
`encode("### " + answer) + [eos]`, and n_markers is 2 for coconut / pause / coconut_no_curriculum and 0 for cot /
no_cot (the reference's `no_special_marker`, which drops <start> and <end>). So labels are -100 on:
  * every question token (including anything `add_special_tokens=True` prepends),
  * <|start-latent|>, every <|latent|>, and <|end-latent|> (the latent markers),
  * pads: the Collator adds -100 on the left pad that aligns the earliest <latent> across the batch and on the right
    pad up to the longest sequence.
Labels equal input_ids on every remaining language step token and on every answer token ("### ", the answer, eos).
The model shifts labels by one inside its loss, as GPT2LMHeadModel does; no shift happens here.

Generation records (for_generation=True, get_question_latent_dataset) are question + markers + latents only and carry
no labels key, exactly as the reference.

Reference semantics that differ from the one-line summary in INTERFACE.md (the reference wins; see the report):
  * Training, stage <= max_latent_stage: n_skip = n_latent = stage even when the example has fewer steps than `stage`
    (the reference pads latents here regardless of pad_latent_to_max). Only for stage > max_latent_stage are all steps
    removed and n_latent = max_latent_stage (pad_latent_to_max) or min(len(steps), max_latent_stage) (otherwise).
  * Generation: n_latent = min(stage, max_latent_stage if pad_latent_to_max else min(max_latent_stage, len(steps))).
  * uniform_prob: the random stage is drawn uniformly from [0, len(steps)] of that example (reference), not from
    [0, max_latent_stage]. rng.random() is consumed once per record for every arm, as the reference consumes
    random.random() per sample.
  * no_cot keeps the reference's literal "skip 100 steps".
  * coconut_no_curriculum has no reference code; the Coconut paper describes it as training directly on the last
    stage's data ("only questions and answers"), so the stage is fixed at max_latent_stage + 1 (all language steps
    removed; n_latent as for any stage beyond max_latent_stage). With pad_latent_to_max and len(steps) <=
    max_latent_stage (ProsQA, ProntoQA) this equals stage max_latent_stage.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field

import torch

SPECIAL_TOKENS = ("<|start-latent|>", "<|end-latent|>", "<|latent|>")  # added in this order, as run.py
ARMS = ("cot", "no_cot", "coconut", "pause", "coconut_no_curriculum")
_CORE_KEYS = ("question", "answer", "steps")


@dataclass
class Example:
    """One Coconut-format record (reference README "Data": question, answer, steps); idx is the position in the file
    after truncation to max_size, as get_dataset assigns it. meta holds every other key (ProsQA: idx_to_symbol,
    edges, root, target, neg_target)."""
    question: str
    answer: str
    steps: list[str]
    idx: int
    meta: dict = field(default_factory=dict)


def load_examples(path: str, max_size: int | None = None) -> list[Example]:
    """Mirrors get_dataset's loading: json.load(open(path))[:max_size], idx = enumerate position."""
    with open(path) as f:
        data = json.load(f)
    if max_size is not None:
        data = data[:max_size]
    return [Example(question=d["question"], answer=d["answer"], steps=list(d["steps"]), idx=i,
                    meta={k: v for k, v in d.items() if k not in _CORE_KEYS})
            for i, d in enumerate(data)]


def add_latent_tokens(tokenizer) -> dict[str, int]:
    """Mirrors run.py: tokenizer.add_tokens for <|start-latent|>, <|end-latent|>, <|latent|> in that order.
    Returns {"start_id", "end_id", "latent_id"}. Resizing the model's embeddings is model.py's job."""
    for t in SPECIAL_TOKENS:
        tokenizer.add_tokens([t])  # one call per token, in order, as run.py
    return {"start_id": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[0]),
            "end_id": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[1]),
            "latent_id": tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[2])}


def _fields(ex, pos):
    if isinstance(ex, Example):
        return ex.question, ex.steps, ex.answer, ex.idx
    return ex["question"], ex["steps"], ex["answer"], ex.get("idx", pos)


def tokenize_examples(examples, tokenizer) -> list[dict]:
    """Mirrors get_dataset.tokenize_sample plus its first-record verification.

    question_tokenized = encode(question + "\\n", add_special_tokens=True); steps_tokenized = [encode(s + "\\n")];
    answer_tokenized = encode("### " + answer) + [eos]. Raises ValueError if the first record's pieces do not
    concatenate to the tokenization of the complete text (the reference asserts the same)."""
    out = []
    for pos, ex in enumerate(examples):
        q, steps, a, idx = _fields(ex, pos)
        out.append({
            "question_tokenized": tokenizer.encode(q + "\n", add_special_tokens=True),
            "steps_tokenized": [tokenizer.encode(s + "\n", add_special_tokens=False) for s in steps],
            "answer_tokenized": tokenizer.encode("### " + a, add_special_tokens=False) + [tokenizer.eos_token_id],
            "idx": idx,
        })
    if out:
        q, steps, a, _ = _fields(examples[0], 0)
        complete = q + "\n" + "\n".join(steps) + "\n### " + a
        complete_tok = tokenizer.encode(complete, add_special_tokens=True) + [tokenizer.eos_token_id]
        d = out[0]
        pieces = d["question_tokenized"] + list(itertools.chain.from_iterable(d["steps_tokenized"])) \
            + d["answer_tokenized"]
        if complete_tok != pieces:
            raise ValueError("tokenize_examples: piecewise tokenization of record 0 differs from the complete text "
                             "(the reference get_dataset asserts this)")
    return out


def _effective_stage(arm: str, stage: int, max_latent_stage: int) -> int:
    """run.py: scheduled_stage = 0 for cot / no_cot; coconut_no_curriculum is pinned to the final stage."""
    if arm in ("cot", "no_cot"):
        return 0
    if arm == "coconut_no_curriculum":
        return max_latent_stage + 1
    return stage


def build_stage_dataset(tok_examples, *, stage: int, arm: str, c_thought: int, max_latent_stage: int,
                        pad_latent_to_max: bool, uniform_prob: float, latent_id: int, start_id: int, end_id: int,
                        rng: random.Random, for_generation: bool = False) -> list[dict]:
    """Mirrors get_cot_latent_dataset.process_dataset (training, validation loss) and
    get_question_latent_dataset.process_dataset (for_generation=True), with run.py's arm flags:
    cot -> cot=True, no_special_marker; no_cot -> no_cot=True, no_special_marker; coconut and pause -> markers and
    the stage curriculum; coconut_no_curriculum -> markers, stage max_latent_stage + 1. Records are returned in input
    order (shuffling is the caller's job, as the reference shuffles after building). See the module docstring for
    where labels are -100 and for the exact latent counts."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    no_special_marker = arm in ("cot", "no_cot")
    no_cot = arm == "no_cot"
    scheduled_stage = _effective_stage(arm, stage, max_latent_stage)
    out = []
    if for_generation:
        for s in tok_examples:
            cap = max_latent_stage if pad_latent_to_max else min(max_latent_stage, len(s["steps_tokenized"]))
            k = min(cap, scheduled_stage) * c_thought
            tokens = (s["question_tokenized"] + ([] if no_special_marker else [start_id]) + [latent_id] * k
                      + ([] if no_special_marker else [end_id]))
            out.append({"input_ids": tokens, "idx": s["idx"], "attention_mask": [1] * len(tokens),
                        "position_ids": list(range(len(tokens)))})
        return out

    n_additional = 0 if no_special_marker else 2
    for s in tok_examples:
        n_steps = len(s["steps_tokenized"])
        if rng.random() < uniform_prob:
            stage_to_train = rng.choice(list(range(n_steps + 1)))
        else:
            stage_to_train = scheduled_stage
        if stage_to_train > max_latent_stage:
            n_skip = 10000
            n_latent = max_latent_stage if pad_latent_to_max else min(n_steps, max_latent_stage)
        else:
            n_skip, n_latent = stage_to_train, stage_to_train
        if no_cot:
            n_skip, n_latent = 100, 0
        n_latent *= c_thought
        q = s["question_tokenized"]
        tokens = (q + ([] if no_special_marker else [start_id]) + [latent_id] * n_latent
                  + ([] if no_special_marker else [end_id])
                  + list(itertools.chain.from_iterable(s["steps_tokenized"][n_skip:])) + s["answer_tokenized"])
        n_masked = len(q) + n_latent + n_additional
        out.append({"input_ids": tokens, "labels": [-100] * n_masked + tokens[n_masked:],
                    "attention_mask": [1] * len(tokens), "idx": s["idx"], "position_ids": list(range(len(tokens)))})
    return out


class Collator:
    """Mirrors MyCollator (+ the tokenizer.pad it calls with padding_side="right").

    If any feature contains latent_id, each feature that does is left-padded with (latest earliest-latent index -
    its earliest-latent index) pads so that the earliest <latent> sits at the same column across the batch (better
    KV-cache reuse); features without latents get no left pad. Left pads get input pad_token_id, attention 0,
    label label_pad and position 0, and the position_ids of such a feature are rewritten to 0 * n_pad + arange(len).
    Then every sequence is right-padded to the longest: input pad_token_id, attention 0, label label_pad,
    position 0. Returns int64 tensors input_ids, attention_mask, position_ids, idx, and labels when the features
    carry labels (generation features do not). Input features are not modified."""

    def __init__(self, pad_token_id: int, latent_id: int | None, label_pad: int = -100):
        self.pad_token_id = pad_token_id
        self.latent_id = latent_id
        self.label_pad = label_pad

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        feats = [{k: (list(v) if isinstance(v, list) else v) for k, v in f.items()} for f in features]
        earliest = [f["input_ids"].index(self.latent_id) for f in feats if self.latent_id in f["input_ids"]]
        if earliest:
            latest = max(earliest)
            for f in feats:
                n = latest - f["input_ids"].index(self.latent_id) if self.latent_id in f["input_ids"] else 0
                f["position_ids"] = [0] * n + list(range(len(f["input_ids"])))
                f["input_ids"] = [self.pad_token_id] * n + f["input_ids"]
                if "labels" in f:
                    f["labels"] = [self.label_pad] * n + f["labels"]
                f["attention_mask"] = [0] * n + f["attention_mask"]
        T = max(len(f["input_ids"]) for f in feats)
        batch = {
            "input_ids": torch.tensor([f["input_ids"] + [self.pad_token_id] * (T - len(f["input_ids"]))
                                      for f in feats], dtype=torch.int64),
            "attention_mask": torch.tensor([f["attention_mask"] + [0] * (T - len(f["attention_mask"]))
                                           for f in feats], dtype=torch.int64),
        }
        if "idx" in feats[0]:
            batch["idx"] = torch.tensor([f["idx"] for f in feats], dtype=torch.int64)
        if "labels" in feats[0] and not all(f["labels"] is None for f in feats):
            L = max(len(f["labels"]) for f in feats)
            batch["labels"] = torch.tensor([f["labels"] + [self.label_pad] * (L - len(f["labels"])) for f in feats],
                                           dtype=torch.int64)
        if "position_ids" in feats[0]:
            P = max(len(f["position_ids"]) for f in feats)
            batch["position_ids"] = torch.tensor([f["position_ids"] + [0] * (P - len(f["position_ids"]))
                                                 for f in feats], dtype=torch.int64)
        return batch


def answer_extract(text: str) -> str:
    """Mirrors run.py: answer_output = text_output.split("#")[-1].replace(",", "").strip(). `text` is the decoded
    generation with skip_special_tokens=True (question included, as the reference decodes outputs[0])."""
    return text.split("#")[-1].replace(",", "").strip()


def answer_normalize(ans: str) -> str:
    """Mirrors run.py: answers_val = d["answer"].replace(",", "").strip(); correct iff answer_extract(...) == this."""
    return ans.replace(",", "").strip()


def cot_extract(text: str) -> str:
    """Mirrors run.py: cot_output = ("\\n".join(text.split("\\n")[1:])).split("#")[0].strip(); compared with
    "\\n".join(steps) for the CoT exact-match metric."""
    return ("\n".join(text.split("\n")[1:])).split("#")[0].strip()


__all__ = ["Example", "load_examples", "add_latent_tokens", "tokenize_examples", "build_stage_dataset", "Collator",
           "answer_extract", "answer_normalize", "cot_extract", "SPECIAL_TOKENS", "ARMS"]
