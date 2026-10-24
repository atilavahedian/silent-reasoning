"""Test oracle: a trimmed port of facebookresearch/coconut dataset.py (MIT, Copyright (c) Meta Platforms, Inc. and
affiliates) and of the stage selection in run.py.

Changes from the original, none of which touch the logic: the HF `datasets.Dataset.map` calls are replaced by list
comprehensions in record order, the multi-GPU broadcast branches are dropped, and `pad_without_fast_tokenizer_warning`
is replaced by `_tokenizer_pad`, which does what `tokenizer.pad(padding=True)` does with padding_side="right" for the
keys input_ids and attention_mask (and turns every other key into a tensor). Randomness uses the module-level `random`
exactly as the original (seed it with random.seed before calling). Used only by the tests; never imported by library
code.
"""
import itertools
import json
import random
from dataclasses import dataclass
from typing import Optional

import torch


def get_dataset(path, tokenizer, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    dataset = [tokenize_sample(d) for d in data]

    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset


def _tokenizer_pad(tokenizer, features):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {}
    for k in features[0].keys():
        if k == "input_ids":
            batch[k] = torch.tensor(
                [f[k] + [tokenizer.pad_token_id] * (max_len - len(f[k])) for f in features]
            )
        elif k == "attention_mask":
            batch[k] = torch.tensor([f[k] + [0] * (max_len - len(f[k])) for f in features])
        else:
            batch[k] = torch.tensor([f[k] for f in features])
    return batch


@dataclass
class MyCollator:

    tokenizer: object
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = _tokenizer_pad(self.tokenizer, non_label_position_features)

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):

    def process_dataset(sample):

        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)

        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return [process_dataset(s) for s in base_dataset_valid]


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )

        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    assert not shuffle  # the original shuffles after mapping; ordering is the caller's job here
    return [process_dataset(s) for s in base_dataset]


# ---- run.py: per-epoch stage and arm flags ---------------------------------------------------------------------

class Config:
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def arm_configs(arm, c_thought, max_latent_stage, pad_latent_to_max, uniform_prob):
    """The yaml flags that select each arm in run.py (coconut / cot / no_cot); pause uses the coconut data path."""
    return Config({
        "coconut": arm in ("coconut", "coconut_no_curriculum"),
        "cot": arm == "cot",
        "no_cot": arm == "no_cot",
        "no_thoughts": False,
        "c_thought": c_thought,
        "max_latent_stage": max_latent_stage,
        "pad_latent_to_max": pad_latent_to_max,
        "uniform_prob": uniform_prob,
    })


def scheduled_stage(configs, epoch, epochs_per_stage):
    """run.py: scheduled_stage = 0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage."""
    return 0 if (configs.cot or configs.no_cot) else epoch // epochs_per_stage


def no_special_marker(configs):
    """run.py: no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts."""
    return configs.cot or configs.no_cot or configs.no_thoughts
