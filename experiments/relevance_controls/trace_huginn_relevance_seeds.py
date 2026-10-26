import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--groups", type=int, default=2)
parser.add_argument("--seed-base", type=int, required=True)
args = parser.parse_args()

MODEL = Path("/mnt/home/user16/nihad/huginn_0125_torch24_compat")
DATA = Path("relevance_control_20261011_171002/diagnostic.json")
assert MODEL.is_dir(), f"Missing model folder: {MODEL}"

audit_path = DATA.parent / "huginn_token_audit.json"
audit = json.loads(audit_path.read_text())
assert audit["summary"]["all_matched"], "Token audit failed"
assert audit["summary"]["dataset_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()

assert torch.cuda.is_available()

data = json.loads(DATA.read_text())
groups = sorted({r["base_id"] for r in data})[:args.groups]
assert len(groups) == args.groups
data = [r for r in data if r["base_id"] in groups]
assert len(data) == args.groups * 10

OUT = Path("mentor_state_comparison") / (
    f"huginn_relevance_seed_{args.seed_base}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(parents=True, exist_ok=False)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL, local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

STEPS = int(model.config.mean_recurrence)
assert getattr(model.config, "test_time_noise", 0) == 0

prepared = []
for item in data:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prepared.append((item, ids))

captured = []
expected_previous = None
maximum_boundary_error = 0.0
original_core = model.core_block_forward

def traced_core(*positional, **kwargs):
    global expected_previous, maximum_boundary_error
    incoming = positional[0][0, -1].detach().cpu().float()
    error = (incoming - expected_previous).abs().max().item()
    maximum_boundary_error = max(maximum_boundary_error, error)
    assert error <= 1e-6, f"Boundary mismatch: {error}"

    result = original_core(*positional, **kwargs)
    vector = result[0][0, -1].detach().cpu().float().clone()
    assert torch.isfinite(vector).all()
    captured.append(vector)
    expected_previous = vector
    return result

traces = []
maximum_final_error = 0.0
instrumentation_check = None

with torch.no_grad():
    for group_index, group in enumerate(groups):
        subset = [
            (item, ids) for item, ids in prepared
            if item["base_id"] == group
        ]
        assert len(subset) == 10
        maximum_length = max(len(ids) for _, ids in subset)
        seed = args.seed_base + group_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        template = model.initialize_state(torch.empty(
            (1, maximum_length, model.config.n_embd),
            device="cuda", dtype=torch.bfloat16,
        ))
        anchor = template[:, -1:, :].clone()

        for item, token_ids in subset:
            ids = torch.tensor([token_ids], device="cuda")
            initial = template[:, :len(token_ids), :].clone()
            initial[:, -1:, :] = anchor

            captured.clear()
            expected_previous = initial[0, -1].cpu().float().clone()
            model.core_block_forward = traced_core

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            try:
                output = model(
                    input_ids=ids,
                    input_states=initial,
                    num_steps=STEPS,
                    use_cache=False,
                )
            finally:
                model.core_block_forward = original_core

            assert len(captured) == STEPS
            states = torch.stack(captured)
            final_normalized = output.latent_states[0, -1].cpu().float()
            reconstructed = model.transformer.ln_f(
                states[-1].to(
                    device="cuda", dtype=torch.bfloat16,
                ).reshape(1, 1, -1)
            )[0, 0].cpu().float()

            error = (reconstructed - final_normalized).abs().max().item()
            maximum_final_error = max(maximum_final_error, error)
            assert error <= 1e-6

            if instrumentation_check is None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                baseline = model(
                    input_ids=ids,
                    input_states=initial,
                    num_steps=STEPS,
                    use_cache=False,
                )
                logit_error = (
                    output.logits - baseline.logits
                ).abs().max().item()
                latent_error = (
                    output.latent_states - baseline.latent_states
                ).abs().max().item()
                assert logit_error <= 1e-6
                assert latent_error <= 1e-6
                instrumentation_check = {
                    "maximum_logit_error": logit_error,
                    "maximum_latent_error": latent_error,
                }
                del baseline

            traces.append({
                "base_id": item["base_id"],
                "condition": item["condition"],
                "query_index": item["query_index"],
                "edit_type": item["edit_type"],
                "changed_chain": item["changed_chain"],
                "relevance": item["relevance"],
                "question": item["question"],
                "target": item["answer"],
                "input_ids": token_ids,
                "anchor_token": tokenizer.convert_ids_to_tokens(
                    token_ids[-1]
                ),
                "seed": seed,
                "initial_anchor_state": anchor[0, 0].cpu().float().clone(),
                "iteration_states": states,
                "final_normalized_state": final_normalized.clone(),
            })
            del output, ids, initial

        del template, anchor
        print(f"Traced {len(traces)}/{len(prepared)}", flush=True)

torch.save(traces, OUT / "iteration_traces.pt")
summary = {
    "model": "Original Huginn",
    "model_directory": str(MODEL),
    "checkpoint_variant": "original downloaded weights; no adaptation",
    "seed_base": args.seed_base,
    "base_groups": len(groups),
    "records": len(traces),
    "recurrent_iterations": STEPS,
    "iteration_tensor_shape": list(traces[0]["iteration_states"].shape),
    "maximum_recurrent_boundary_error": maximum_boundary_error,
    "maximum_final_normalization_error": maximum_final_error,
    "instrumentation_check": instrumentation_check,
    "correctness_filter": False,
    "answer_generation": False,
    "base_ids": groups,
    "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    "precision": "bfloat16",
    "state_definition": (
        "Raw state after each recurrent core iteration, "
        "at the final native-chat-prompt position."
    ),
    "initialization_control": (
        "Native initializer; shared positional template within each "
        "base group, with final prompt anchor aligned."
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("Saved:", OUT)
