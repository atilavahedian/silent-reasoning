"""Test oracle: a trimmed port of facebookresearch/coconut coconut.py (MIT, Copyright (c) Meta Platforms, Inc. and
affiliates), class Coconut, methods forward and generate.

Changes from the original, none of which touch the arithmetic:
  * the sliced KV cache is passed as a tuple of (k, v) tuples instead of a list, because transformers >= 4.5x only
    converts a *tuple* legacy cache to DynamicCache (a list is passed through and fails); iterating over the returned
    cache works for both the legacy tuple and DynamicCache (Cache.__iter__ yields (keys, values) per layer);
  * the FSDP `synced_gpus` padding loop and `output_embedding` branch of generate are dropped; generate returns the
    token list;
  * `gen_forward_cnt` bookkeeping is dropped.
Used only by the tests; never imported by library code.
"""
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


def _legacy(kv_cache, n):
    return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in kv_cache)


class Coconut(nn.Module):

    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super(Coconut, self).__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        logits = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache == None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]: next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = _legacy(kv_cache, next_compute_range[0])
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            next_compute_range = (
                next_compute_range[1],
                (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1),
            )
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])]
            )

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]: next_compute_range[1]],
            past_key_values=(_legacy(kv_cache, next_compute_range[0]) if kv_cache else None),
            output_hidden_states=True,
        )
        logits.append(outputs.logits)

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=16, **kwargs):
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
        return tokens
