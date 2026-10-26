import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--groups", type=int, default=2)
args = parser.parse_args()

MODEL = Path("/mnt/home/user16/nihad/ouro_1_4b")
DATA = Path(
    "/mnt/home/user16/nihad/coconut/mentor_experiment/"
    "relevance_control_20261011_171002/diagnostic.json"
)
ROOT = Path(
    "/mnt/home/user16/nihad/coconut/mentor_experiment/"
    "mentor_state_comparison"
)
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"

manifest = json.loads((MODEL / "download_revision.json").read_text())
assert manifest["repo_id"] == "ByteDance/Ouro-1.4B"
assert manifest["revision"] == REVISION

data = json.loads(DATA.read_text())
base_ids = sorted({row["base_id"] for row in data})
assert 1 <= args.groups <= len(base_ids)

selected_ids = base_ids[:args.groups]
selected = [
    row for row in data if row["base_id"] in selected_ids
]

conditions = {
    "original", "path_chain_0", "path_chain_1", "endpoint_chain_0", "endpoint_chain_1"
}
expected = {(condition, query) for condition in conditions for query in (0, 1)}

for base_id in selected_ids:
    rows = [row for row in selected if row["base_id"] == base_id]
    assert len(rows) == 10
    assert {
        (row["condition"], row["query_index"]) for row in rows
    } == expected

assert torch.cuda.is_available(), "CUDA unavailable"
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
)
assert tokenizer.chat_template, "Native chat template not found"

model, loading = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    output_loading_info=True,
)

for key in (
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
):
    assert not loading.get(key), (key, loading[key])

model.to("cuda").eval()
core = model.model
loops = int(core.total_ut_steps)

assert loops == 4
assert model.config.hidden_size == 2048

print("MODEL LOADED", flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Loops:", loops, flush=True)
print("Precision:", next(model.parameters()).dtype, flush=True)

traces = []
max_errors = {
    "instrumentation_logit_error": 0.0,
    "raw_boundary_error": 0.0,
    "next_loop_input_error": 0.0,
    "returned_state_error": 0.0,
    "final_state_error": 0.0,
}

def anchor(tensor):
    return tensor[0, -1].detach().float().cpu().clone()

def error(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return float((a - b).abs().max())

with torch.inference_mode():
    for index, item in enumerate(selected):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        kwargs = dict(
            **inputs,
            use_cache=False,
            logits_to_keep=1,
            exit_at_step=loops - 1,
        )

        reference = model(**kwargs).logits.detach().float().cpu()

        captured = {
            "raw": [],
            "normalized": [],
            "loop_inputs": [],
            "last_block": [],
        }
        returned = {}

        def norm_pre(module, values):
            captured["raw"].append(anchor(values[0]))

        def norm_post(module, values, output):
            captured["normalized"].append(anchor(output))

        def first_pre(module, values, keyword_values):
            state = keyword_values.get("hidden_states")
            if state is None:
                state = values[0]
            captured["loop_inputs"].append(anchor(state))

        def last_post(module, values, output):
            captured["last_block"].append(anchor(output))

        def core_post(module, values, output):
            base_output, states, gates = output
            returned["states"] = torch.stack(
                [anchor(state) for state in states]
            )
            returned["final"] = anchor(base_output.last_hidden_state)

        handles = [
            core.norm.register_forward_pre_hook(norm_pre),
            core.norm.register_forward_hook(norm_post),
            core.layers[0].register_forward_pre_hook(
                first_pre, with_kwargs=True
            ),
            core.layers[-1].register_forward_hook(last_post),
            core.register_forward_hook(core_post),
        ]

        try:
            traced = model(**kwargs).logits.detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()

        for name, values in captured.items():
            assert len(values) == loops, (name, len(values))

        raw = torch.stack(captured["raw"])
        normalized = torch.stack(captured["normalized"])
        loop_inputs = torch.stack(captured["loop_inputs"])
        last_block = torch.stack(captured["last_block"])

        assert raw.shape == (loops, model.config.hidden_size)
        assert torch.isfinite(raw).all()
        assert torch.isfinite(normalized).all()

        checks = {
            "instrumentation_logit_error": error(reference, traced),
            "raw_boundary_error": error(raw, last_block),
            "next_loop_input_error": error(
                normalized[:-1], loop_inputs[1:]
            ),
            "returned_state_error": error(
                normalized, returned["states"]
            ),
            "final_state_error": error(
                normalized[-1], returned["final"]
            ),
        }

        for name, value in checks.items():
            max_errors[name] = max(max_errors[name], value)
            assert value == 0.0, (item["base_id"], name, value)

        traces.append({
            "base_id": item["base_id"],
            "condition": item["condition"],
            "query_index": item["query_index"],
            "question": item["question"],
            "prompt": prompt,
            "input_ids": inputs["input_ids"][0].cpu().tolist(),
            "anchor_position": inputs["input_ids"].shape[1] - 1,
            "raw_vectors": raw,
            "normalized_vectors": normalized,
        })

        if (index + 1) % 8 == 0:
            print(
                f"Traced {index + 1}/{len(selected)}",
                flush=True,
            )

summary = {
    "model": "Original Ouro-1.4B",
    "revision": REVISION,
    "checkpoint_variant": "original weights; no synthetic adaptation",
    "base_groups": args.groups,
    "records": len(selected),
    "recurrent_loops": loops,
    "iteration_tensor_shape": [loops, model.config.hidden_size],
    "extraction_checks": max_errors,
    "correctness_filter": False,
    "answer_generation": False,
    "base_ids": selected_ids,
    "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    "precision": str(next(model.parameters()).dtype),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "attention_backend": "eager",
    "use_cache": False,
    "state_definition": (
        "Final native-chat-prompt position at each loop boundary. "
        "Raw states precede the shared final RMSNorm; normalized "
        "states are passed into the next loop."
    ),
    "readout_loop": loops,
    "code_sha256": {
        name: hashlib.sha256((MODEL / name).read_bytes()).hexdigest()
        for name in ("modeling_ouro.py", "configuration_ouro.py")
    },
}

out = ROOT / (
    "ouro_relevance_control_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
out.mkdir(parents=True, exist_ok=False)

torch.save(traces, out / "latent_traces.pt")
(out / "summary.json").write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2), flush=True)
print("Saved:", out, flush=True)
