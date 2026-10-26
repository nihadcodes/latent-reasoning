# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel


Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "logits"]
)

MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()

        self.gen_forward_cnt = 0

        # Validation trace for intermediate latent states
        self.latent_trace = []

        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # Tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = (
                self.base_causallm.transformer.get_input_embeddings()
            )
        else:
            self.embedding = (
                self.base_causallm.get_input_embeddings()
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        **kwargs
    ):

        # Reset trace for this forward pass
        self.latent_trace = []

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()

        # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [
                idx[1].item()
                for idx in latent_indices
                if idx[0] == i
            ]
            for i in range(input_ids.shape[0])
        ]

        # bs, num_latent_tokens_in_the_instance
        # (difference across the batch)

        max_n_latents = max(
            [len(latent_list) for latent_list in latent_lists]
        )

        next_compute_range = (
            0,
            input_ids.shape[1]
        )

        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:

            next_compute_range = (
                0,
                latent_indices[:, 1].min().item()
            )

            # Before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache is None:

                # First forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :,
                        next_compute_range[0]:next_compute_range[1],
                        :
                    ],
                    attention_mask=attention_mask[
                        :,
                        next_compute_range[0]:next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :,
                        next_compute_range[0]:next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )

                hidden_states_offset = 0

            else:

                # Extract KV cache to reuse
                past_key_values = [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :,
                        next_compute_range[0]:next_compute_range[1],
                        :
                    ],
                    attention_mask=attention_mask[
                        :,
                        :next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :,
                        next_compute_range[0]:next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

                # When we use KV cache for the first k tokens,
                # in outputs.hidden_states, [0, k) will be skipped.
                # Therefore, we need to keep this offset to
                # correctly use the hidden states.

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            print(
                f"[ShapeDebug] "
                f"pass={pass_idx}, "
                f"hidden_states_shape={tuple(hidden_states.shape)}, "
                f"hidden_states_offset={hidden_states_offset}, "
                f"compute_range={next_compute_range}"
            )

            # Get the last layer hidden states

            kv_cache = outputs.past_key_values

            # -------------------------------------------------
            # Feedback the continuous thoughts to input_embeds
            # -------------------------------------------------

            # First decide the positions to provide feedback
            filling_indices = [
                (
                    instance_idx,
                    mask_list[pass_idx]
                )
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Avoid in-place operations.
            #
            # Break down inputs_embeds:
            # (batch_size, sequence_length, hidden_size)
            #
            # Into a list of lists containing 1-D tensors.

            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # -------------------------------------------------
            # Replace latent tokens with continuous thoughts
            # -------------------------------------------------

            for idx_pair in filling_indices:

                batch_idx, token_idx = idx_pair

                # Extract the hidden state used for latent injection
                source_position = (
                    token_idx - 1 - hidden_states_offset
                )

                # Validate source position before indexing
                if (
                    source_position < 0
                    or source_position >= hidden_states.shape[1]
                ):

                    raise RuntimeError(
                        f"Invalid source_position={source_position}. "
                        f"hidden_states.shape={hidden_states.shape}, "
                        f"token_idx={token_idx}, "
                        f"hidden_states_offset={hidden_states_offset}, "
                        f"pass_idx={pass_idx}"
                    )

                latent_vector = hidden_states[
                    batch_idx,
                    source_position,
                    :
                ]

                # Validate the extracted vector dimension
                if (
                    latent_vector.shape[-1]
                    != inputs_embeds.shape[-1]
                ):

                    raise RuntimeError(
                        f"Dimension mismatch: "
                        f"latent_vector.shape={latent_vector.shape}, "
                        f"inputs_embeds.shape={inputs_embeds.shape}"
                    )

                # Record the exact vector and its metadata
                self.latent_trace.append(
                    {
                        "pass_idx": int(pass_idx),
                        "batch_idx": int(batch_idx),
                        "latent_position": int(token_idx),
                        "source_position": int(source_position),
                        "hidden_states_offset": int(
                            hidden_states_offset
                        ),
                        "vector_shape": list(
                            latent_vector.shape
                        ),
                        "vector_norm": float(
                            torch.norm(latent_vector)
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "latent_vector": (
                            latent_vector.detach()
                            .cpu()
                            .clone()
                        ),
                    }
                )

                # Inject the extracted vector into the latent position
                tensor_list[batch_idx][token_idx] = latent_vector

                # Validate that the vector was inserted correctly
                injected_vector = (
                    tensor_list[batch_idx][token_idx]
                )

                injection_error = torch.max(
                    torch.abs(
                        injected_vector - latent_vector
                    )
                ).item()

                if injection_error > 1e-6:

                    raise RuntimeError(
                        f"Latent injection validation failed: "
                        f"error={injection_error}, "
                        f"pass_idx={pass_idx}, "
                        f"batch_idx={batch_idx}, "
                        f"token_idx={token_idx}"
                    )

                print(
                    f"[Validation] "
                    f"pass={pass_idx}, "
                    f"batch={batch_idx}, "
                    f"latent_position={token_idx}, "
                    f"source_position={source_position}, "
                    f"injection_error={injection_error:.8f}"
                )

            # -------------------------------------------------
            # Assemble the new inputs_embeds
            # -------------------------------------------------

            inputs_embeds = torch.stack(
                [
                    torch.stack(
                        tensor_list[batch_idx]
                    )
                    for batch_idx in range(
                        inputs_embeds.shape[0]
                    )
                ]
            )

        # -----------------------------------------------------
        # Final forward pass
        # -----------------------------------------------------

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :,
                next_compute_range[0]:next_compute_range[1],
                :
            ],
            attention_mask=attention_mask[
                :,
                :next_compute_range[1]
            ],
            position_ids=position_ids[
                :,
                next_compute_range[0]:next_compute_range[1]
            ],
            past_key_values=(
                [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(
            logits,
            dim=-2
        )

        shift_logits = logits[
            ...,
            :-1,
            :
        ].contiguous()

        shift_labels = labels[
            ...,
            1:
        ].contiguous()

        loss_fct = CrossEntropyLoss()

        loss = loss_fct(
            shift_logits.view(
                -1,
                shift_logits.size(-1)
            ),
            shift_labels.view(-1)
        )

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits
        )

    def train(self):

        self.base_causallm.train()

    def eval(self):

        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert (
            input_ids.shape[0] == 1
        ), "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()

        # Placeholder; not used

        outputs = self.forward(
            input_ids,
            torch.ones_like(
                input_ids,
                device=input_ids.device
            ),
            labels,
            torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device
            ).reshape(1, -1)
        )

        inputs_embeds = outputs.inputs_embeds

        # Get the first token using the current hidden state
        next_token = torch.argmax(
            outputs.logits[0, -1]
        ).item()

        tokens.append(next_token)

        new_token_embed = self.embedding(
            torch.tensor(
                next_token,
                device=input_ids.device
            )
        ).view(1, 1, -1)

        new_inputs_embeds = torch.cat(
            (
                inputs_embeds,
                new_token_embed
            ),
            dim=1
        )

        # Get other tokens
        for _ in range(max_new_tokens - 1):

            outputs = self.base_causallm(
                inputs_embeds=new_inputs_embeds
            )

            self.gen_forward_cnt += 1

            next_token = torch.argmax(
                outputs.logits[0, -1]
            ).item()

            if next_token == self.eos_token_id:
                break

            tokens.append(next_token)

            new_token_embed = self.embedding(
                torch.tensor(
                    next_token,
                    device=input_ids.device
                )
            ).view(1, 1, -1)

            new_inputs_embeds = torch.cat(
                (
                    new_inputs_embeds,
                    new_token_embed
                ),
                dim=1
            )

        if synced_gpus:

            while (
                self.gen_forward_cnt
                < max_new_tokens + MAX_N_LATENT
            ):

                self.gen_forward_cnt += 1

                _ = self.base_causallm(
                    inputs_embeds=new_inputs_embeds
                )

        if output_embedding:

            return (
                torch.tensor(tokens).view(1, -1),
                new_inputs_embeds
            )

        else:

            return torch.tensor(tokens).view(1, -1)