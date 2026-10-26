import csv
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

COCONUT = Path(
    "mentor_state_comparison/coconut_original_20261005_164839_863293"
)
HUGINN = Path(
    "mentor_state_comparison/huginn_original_20261005_170849_340005"
)
OUT = Path("mentor_state_comparison") / (
    "analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(exist_ok=False)

c_summary = json.loads((COCONUT / "summary.json").read_text())
h_summary = json.loads((HUGINN / "summary.json").read_text())
assert c_summary["dataset_sha256"] == h_summary["dataset_sha256"]
assert c_summary["base_ids"] == h_summary["base_ids"]

groups = c_summary["base_ids"]
conditions = ("original", "relabeling", "path_change", "answer_change")
assert len(groups) == 40

def key(r):
    return (r["base_id"], r["condition"], int(r["query_index"]))

def load(path):
    rows = torch.load(path, map_location="cpu", weights_only=True)
    indexed = {key(r): r for r in rows}
    assert len(indexed) == len(rows) == 320
    return indexed

c = load(COCONUT / "layer_traces.pt")
h = load(HUGINN / "iteration_traces.pt")
expected = {
    (g, condition, query)
    for g in groups
    for condition in conditions
    for query in (0, 1)
}
assert set(c) == set(h) == expected

for k in expected:
    assert c[k]["question"] == h[k]["question"], k
    assert c[k]["target"] == h[k]["target"], k
    assert c[k]["injected_vectors"].shape == (6, 768)
    assert c[k]["raw_block_outputs"].shape == (6, 12, 768)
    assert h[k]["iteration_states"].shape == (32, 5280)

# Verify the shared Huginn anchor initialization within every base group.
for g in groups:
    reference = h[(g, "original", 0)]["initial_anchor_state"]
    for condition in conditions:
        for query in (0, 1):
            assert torch.equal(
                reference,
                h[(g, condition, query)]["initial_anchor_state"],
            )

generator = torch.Generator().manual_seed(2026)
bootstrap_indices = torch.randint(
    len(groups), (2000, len(groups)), generator=generator,
)

def estimate(values):
    values = values.double()
    assert values.shape == (40,)
    assert torch.isfinite(values).all()
    means = values[bootstrap_indices].mean(dim=1)
    return (
        values.mean().item(),
        torch.quantile(means, 0.025).item(),
        torch.quantile(means, 0.975).item(),
    )

def add_estimate(row, name, values):
    mean, low, high = estimate(values)
    row[name + "_mean"] = mean
    row[name + "_ci_low"] = low
    row[name + "_ci_high"] = high

def write_csv(name, rows):
    assert rows
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

models = {
    "Coconut": (c, "injected_vectors"),
    "Huginn": (h, "iteration_states"),
}
paired_rows = []
dynamics_rows = []

for model_name, (records, field) in models.items():
    for query in (0, 1):
        original = torch.stack([
            records[(g, "original", query)][field].float()
            for g in groups
        ])
        assert torch.isfinite(original).all()
        assert (original.norm(dim=-1) > 0).all()

        for condition in conditions[1:]:
            variant = torch.stack([
                records[(g, condition, query)][field].float()
                for g in groups
            ])
            assert torch.isfinite(variant).all()
            assert (variant.norm(dim=-1) > 0).all()
            cosine = F.cosine_similarity(original, variant, dim=-1)
            displacement = (
                (variant - original).norm(dim=-1)
                / original.norm(dim=-1).clamp_min(1e-12)
            )

            for step in range(original.shape[1]):
                row = {
                    "model": model_name,
                    "condition": condition,
                    "query_index": query,
                    "step": step + 1,
                    "base_groups": 40,
                }
                add_estimate(row, "cosine", cosine[:, step])
                add_estimate(
                    row, "normalized_displacement",
                    displacement[:, step],
                )
                paired_rows.append(row)

        for condition in conditions:
            states = torch.stack([
                records[(g, condition, query)][field].float()
                for g in groups
            ])
            norms = states.norm(dim=-1)
            assert torch.isfinite(states).all()
            assert (norms > 0).all()

            for step in range(states.shape[1]):
                row = {
                    "model": model_name,
                    "condition": condition,
                    "query_index": query,
                    "step": step + 1,
                    "base_groups": 40,
                }
                add_estimate(row, "state_norm", norms[:, step])

                for metric in ("previous_cosine", "previous_displacement"):
                    for suffix in ("mean", "ci_low", "ci_high"):
                        row[metric + "_" + suffix] = None

                if step > 0:
                    previous = states[:, step - 1]
                    current = states[:, step]
                    add_estimate(
                        row, "previous_cosine",
                        F.cosine_similarity(previous, current, dim=-1),
                    )
                    add_estimate(
                        row, "previous_displacement",
                        (current - previous).norm(dim=-1)
                        / previous.norm(dim=-1).clamp_min(1e-12),
                    )
                dynamics_rows.append(row)

# Raw Coconut block outputs: final normalization is NOT mixed into layer 12.
layer_rows = []
for query in (0, 1):
    original = torch.stack([
        c[(g, "original", query)]["raw_block_outputs"].float()
        for g in groups
    ])
    for condition in conditions[1:]:
        variant = torch.stack([
            c[(g, condition, query)]["raw_block_outputs"].float()
            for g in groups
        ])
        assert torch.isfinite(original).all()
        assert torch.isfinite(variant).all()
        assert (original.norm(dim=-1) > 0).all()
        assert (variant.norm(dim=-1) > 0).all()
        cosine = F.cosine_similarity(original, variant, dim=-1)
        displacement = (
            (variant - original).norm(dim=-1)
            / original.norm(dim=-1).clamp_min(1e-12)
        )

        for step in range(6):
            for layer in range(12):
                row = {
                    "condition": condition,
                    "query_index": query,
                    "latent_step": step + 1,
                    "block": layer + 1,
                    "base_groups": 40,
                }
                add_estimate(row, "cosine", cosine[:, step, layer])
                add_estimate(
                    row, "normalized_displacement",
                    displacement[:, step, layer],
                )
                layer_rows.append(row)

write_csv("paired_prompt_similarity.csv", paired_rows)
write_csv("within_prompt_dynamics.csv", dynamics_rows)
write_csv("coconut_layer_comparison.csv", layer_rows)

summary = {
    "base_groups": 40,
    "records_per_model": 320,
    "paired_prompt_rows": len(paired_rows),
    "dynamics_rows": len(dynamics_rows),
    "coconut_layer_rows": len(layer_rows),
    "bootstrap_replicates": 2000,
    "bootstrap_unit": "base_group",
    "interval_type": "pointwise percentile 95%; not simultaneous",
    "correctness_filter": False,
    "dataset_sha256": c_summary["dataset_sha256"],
    "coconut_source": str(COCONUT),
    "huginn_source": str(HUGINN),
    "limitations": [
        "Different checkpoints, architectures, precision and native prompts.",
        "Coconut moving injection-source position versus Huginn fixed prompt anchor.",
        "Huginn shared initialization control; one realization per base group.",
        "Relabeling changes symbol identities; raw cosine is not semantic invariance.",
        "No direct cross-model vector cosine and no matched-compute claim.",
    ],
}
(OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "coconut_trace_summary.json").write_text(
    json.dumps(c_summary, indent=2)
)
(OUT / "huginn_trace_summary.json").write_text(
    json.dumps(h_summary, indent=2)
)

archive = OUT / "original_model_state_analysis.zip"
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for path in sorted(OUT.iterdir()):
        if path.suffix in (".csv", ".json"):
            z.write(path, arcname=path.name)
    z.write(Path(__file__), arcname=Path(__file__).name)

print(json.dumps(summary, indent=2))
print("\nFINAL-STEP PAIRED SIMILARITY")
for row in paired_rows:
    final_step = 6 if row["model"] == "Coconut" else 32
    if row["step"] == final_step:
        print(
            f'{row["model"]:7s} {row["condition"]:14s} '
            f'query={row["query_index"]} '
            f'cosine={row["cosine_mean"]:.6f} '
            f'displacement={row["normalized_displacement_mean"]:.6f}'
        )
print("\nUPLOAD THIS ZIP:", archive.resolve())
