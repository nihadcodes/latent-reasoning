import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path("..").resolve()))
from coconut import Coconut

parser = argparse.ArgumentParser()
parser.add_argument("--groups", type=int, default=2)
args = parser.parse_args()

DATA = Path("relevance_control_20261011_171002/diagnostic.json")
CHECKPOINT = Path("../checkpoints/checkpoint_24")
BASE = Path("../gpt2_model")
STEPS = 6


audit_path = DATA.parent / "coconut_token_audit.json"
audit = json.loads(audit_path.read_text())
assert audit["summary"]["all_matched"], "Token audit failed"
assert audit["summary"]["dataset_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()

assert torch.cuda.is_available()
assert CHECKPOINT.is_file()
records = json.loads(DATA.read_text())
group_ids = sorted({r["base_id"] for r in records})[:args.groups]
assert len(group_ids) == args.groups
records = [r for r in records if r["base_id"] in group_ids]
assert len(records) == args.groups * 10

OUT = Path("mentor_state_comparison") / (
    "coconut_relevance_control_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(parents=True, exist_ok=False)

tokenizer = AutoTokenizer.from_pretrained(
    BASE, local_files_only=True,
)
tokenizer.add_tokens(
    ["<|start-latent|>", "<|end-latent|>", "<|latent|>"],
    special_tokens=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE, local_files_only=True,
)
base.resize_token_embeddings(len(tokenizer))

start = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end = tokenizer.convert_tokens_to_ids("<|end-latent|>")
latent = tokenizer.convert_tokens_to_ids("<|latent|>")
model = Coconut(base, latent, start, end, tokenizer.eos_token_id)

weights = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True,
)
model.load_state_dict(weights, strict=True)
del weights
model.to("cuda")
model.eval()

# Store only relevant positions, not full prompt activations.
passes = []
active = {}

def begin_pass(module, positional, kwargs):
    active.clear()
    embeddings = kwargs["inputs_embeds"]
    active["first_input"] = embeddings[0, 0].detach().cpu().clone()
    active["last_input"] = embeddings[0, -1].detach().cpu().clone()
    active["blocks"] = {}

def block_hook(index):
    def capture(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        active["blocks"][index] = (
            hidden[0, -1].detach().cpu().clone()
        )
    return capture

def finish_pass(module, inputs, output):
    assert len(active["blocks"]) == len(base.transformer.h)
    passes.append({
        "first_input": active["first_input"],
        "last_input": active["last_input"],
        "block_outputs": torch.stack([
            active["blocks"][i]
            for i in range(len(base.transformer.h))
        ]),
        "final_normalized": (
            output.hidden_states[-1][0, -1]
            .detach().cpu().clone()
        ),
    })

handles = [
    base.register_forward_pre_hook(begin_pass, with_kwargs=True),
    base.register_forward_hook(finish_pass),
]
handles += [
    block.register_forward_hook(block_hook(i))
    for i, block in enumerate(base.transformer.h)
]

traces = []
max_feedback_error = 0.0
max_final_state_error = 0.0

try:
    with open(os.devnull, "w") as quiet, torch.no_grad():
        for index, item in enumerate(records):
            passes.clear()
            torch.manual_seed(2026)
            torch.cuda.manual_seed_all(2026)

            ids = (
                tokenizer.encode(
                    item["question"] + "\n",
                    add_special_tokens=True,
                )
                + [start] + [latent] * STEPS + [end]
            )
            inputs = torch.tensor([ids], device="cuda")

            with contextlib.redirect_stdout(quiet):
                output = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    labels=inputs.clone(),
                    position_ids=torch.arange(
                        len(ids), device="cuda",
                    ).unsqueeze(0),
                )

            assert len(passes) == STEPS + 1
            assert len(model.latent_trace) == STEPS
            injected = torch.stack([
                r["latent_vector"] for r in model.latent_trace
            ])
            raw_blocks = torch.stack([
                p["block_outputs"] for p in passes[:STEPS]
            ])
            normalized = torch.stack([
                p["final_normalized"] for p in passes[:STEPS]
            ])
            source_embeddings = torch.stack([
                p["last_input"] for p in passes[:STEPS]
            ])

            assert torch.isfinite(raw_blocks).all()
            assert torch.isfinite(injected).all()

            for step in range(STEPS):
                feedback_error = (
                    injected[step] - passes[step + 1]["first_input"]
                ).abs().max().item()
                final_error = (
                    injected[step] - normalized[step]
                ).abs().max().item()
                max_feedback_error = max(
                    max_feedback_error, feedback_error
                )
                max_final_state_error = max(
                    max_final_state_error, final_error
                )
                assert feedback_error <= 1e-6
                assert final_error <= 1e-6

            traces.append({
                "base_id": item["base_id"],
                "condition": item["condition"],
                "query_index": item["query_index"],
                "edit_type": item["edit_type"],
                "changed_chain": item["changed_chain"],
                "relevance": item["relevance"],
                "question": item["question"],
                "target": item["answer"],
                "input_ids": ids,
                "injected_vectors": injected,
                "raw_block_outputs": raw_blocks,
                "final_normalized_states": normalized,
                "source_input_embeddings": source_embeddings,
            })
            del output, inputs

            if (index + 1) % 8 == 0:
                print(f"Traced {index + 1}/{len(records)}", flush=True)
finally:
    for handle in handles:
        handle.remove()

torch.save(traces, OUT / "layer_traces.pt")
summary = {
    "model": "Coconut",
    "checkpoint": str(CHECKPOINT.resolve()),
    "checkpoint_loaded_strictly": True,
    "checkpoint_variant": "mentor original; no synthetic adaptation",
    "split": "relevance_control_diagnostic",
    "base_groups": len(group_ids),
    "records": len(traces),
    "latent_steps": STEPS,
    "transformer_blocks": len(base.transformer.h),
    "raw_block_tensor_shape": list(
        traces[0]["raw_block_outputs"].shape
    ),
    "maximum_feedback_error": max_feedback_error,
    "maximum_final_state_error": max_final_state_error,
    "correctness_filter": False,
    "answer_generation": False,
    "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    "base_ids": group_ids,
    "precision": str(next(base.parameters()).dtype),
    "torch_version": torch.__version__,
    "state_definition": (
        "For each latent injection, raw outputs of every GPT-2 block "
        "at its source position. Final layer normalization is stored "
        "separately. Steps correspond to vectors injected into slots."
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("Saved:", OUT)
