#!/usr/bin/env python3
"""Fixed-cohort pilot of the RBS notebook's node-embedding readout.

No training, answer generation or correctness filtering. Raw dot product follows
the notebook's analysis; cosine and graph-bootstrap contrasts are diagnostics.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

GROUPS = ("not_reached_up_to_k", "reachable_up_to_k", "walk_frontier",
          "reference_optimal", "globally_unreachable", "reachable_after_k")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_info(item):
    adjacency, reverse = defaultdict(list), defaultdict(list)
    nodes = set()
    for u, v in item["edges"]:
        adjacency[u].append(v)
        reverse[v].append(u)
        nodes.update((u, v))
    nodes.update((item["root"], item["target"], item["neg_target"]))

    def distances(edges, start):
        found, pending = {start: 0}, deque([start])
        while pending:
            u = pending.popleft()
            for v in edges[u]:
                if v not in found:
                    found[v] = found[u] + 1
                    pending.append(v)
        return found

    distance = distances(adjacency, item["root"])
    to_target = distances(reverse, item["target"])
    assert item["target"] in distance, "Target is not reachable"
    assert item["neg_target"] not in distance, "Negative target is reachable"
    assert all(isinstance(v, int) and 0 <= v <= 30 for v in nodes)
    return nodes, adjacency, distance, to_target


def node_groups(item, step):
    nodes, adjacency, distance, to_target = graph_info(item)
    frontier, reached = {item["root"]}, {item["root"]}
    for _ in range(step):
        frontier = {v for u in frontier for v in adjacency[u]}
        reached.update(frontier)
    optimal = set(item["neighbor_k"][str(step)])
    expected = {v for v in nodes if distance.get(v) == step
                and step + to_target.get(v, math.inf) == distance[item["target"]]}
    assert optimal == expected, "Reference annotation disagrees with solver"
    assert optimal <= frontier <= reached <= nodes
    return {
        "not_reached_up_to_k": nodes - reached,
        "reachable_up_to_k": reached,
        "walk_frontier": frontier,
        "reference_optimal": optimal,
        "globally_unreachable": nodes - set(distance),
        "reachable_after_k": set(distance) - reached,
    }


def audit(data, steps):
    eligible = []
    signatures = set()
    for index, item in enumerate(data):
        graph_info(item)
        signature = tuple(sorted(map(tuple, item["edges"])))
        assert signature not in signatures, "Repeated edge set: group before sampling"
        signatures.add(signature)
        for step in range(1, steps + 1):
            if str(step) in item["neighbor_k"]:
                node_groups(item, step)
        if all(str(k) in item["neighbor_k"] for k in range(1, steps + 1)):
            eligible.append(index)
    return eligible


def self_test():
    item = {"edges": [[0, 1], [0, 2], [1, 2], [2, 3], [4, 5]],
            "root": 0, "target": 3, "neg_target": 5,
            "neighbor_k": {"1": [2], "2": [3]}}
    one, two = node_groups(item, 1), node_groups(item, 2)
    assert one["walk_frontier"] == {1, 2}
    assert two["walk_frontier"] == {2, 3}
    assert two["reachable_up_to_k"] == {0, 1, 2, 3}
    assert two["globally_unreachable"] == {4, 5}
    assert one["reachable_after_k"] == {3}
    assert 2 in one["walk_frontier"] & two["walk_frontier"]
    print("GRAPH-DEFINITION SELF-TEST PASSED")


def write_csv(path, rows):
    assert rows
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--graphs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-data", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_data:
        data = json.loads(args.audit_data.read_text())
        pool = audit(data, args.steps)
        print(json.dumps({"records": len(data), "eligible_fixed_cohort": len(pool),
                          "steps": args.steps, "solver_audit_passed": True}, indent=2))
        return

    # Import model dependencies only for the actual GPU run.
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from coconut import Coconut
    from stokenizer import STokenizer

    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert args.graphs > 0 and args.bootstrap >= 100
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = root / "data/prosqa_test_graph_4_coconut.json"
    checkpoint = root / "artifacts/coconut_theory/checkpoint_300"
    expected_hash = "6fafa619ab8b57c065e9f3fe94d3eecb13a42d880441d582b1b0bfa3ae8a6fc0"
    assert sha256(checkpoint) == expected_hash, "Checkpoint checksum mismatch"
    data = json.loads(dataset.read_text())
    pool = audit(data, args.steps)
    assert args.graphs <= len(pool), f"Only {len(pool)} eligible graphs"
    indices = random.Random(args.seed).sample(pool, args.graphs)

    config_path = root / "configs/symbol-2layer-8head-768dim.json"
    tokenizer = STokenizer()
    config = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
    base = AutoModelForCausalLM.from_config(config, attn_implementation="eager").float()
    latent = tokenizer.convert_tokens_to_ids("<|latent|>")
    start = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    eos = tokenizer.convert_tokens_to_ids("<eos>")
    assert config.vocab_size == len(tokenizer) == 40
    model = Coconut(base, latent, start, end, eos)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.to("cuda")
    model.eval()  # The repository's eval() returns None: do not assign it.
    print("STRICT LOADING AND FULL DATASET SOLVER AUDIT PASSED", flush=True)

    def extract(prompt, n, check=False):
        full = prompt + " " + " ".join(["<|latent|>"] * n)
        ids = torch.tensor([tokenizer.encode(full, add_special_tokens=False)], device="cuda")
        assert ids.shape[1] <= config.n_positions
        positions = (ids[0] == latent).nonzero().flatten().tolist()
        assert len(positions) == n
        calls, handles = [], []
        if check:
            def before(module, pargs, kwargs):
                calls.append({"p": kwargs["position_ids"].detach().cpu(),
                              "x": kwargs["inputs_embeds"].detach().cpu()})
            def after(module, pargs, kwargs, output):
                calls[-1]["h"] = output.hidden_states[-1].detach().cpu()
            handles = [base.register_forward_pre_hook(before, with_kwargs=True),
                       base.register_forward_hook(after, with_kwargs=True)]
        try:
            with torch.no_grad():
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               labels=ids.clone(), position_ids=torch.arange(
                                   ids.shape[1], device="cuda").unsqueeze(0))
        finally:
            for handle in handles:
                handle.remove()
        assert torch.isfinite(output.logits).all()
        vectors = output.inputs_embeds[0, positions, :].detach().cpu()
        assert torch.isfinite(vectors).all()
        errors = []
        if check:
            assert len(calls) == n + 1
            for k, position in enumerate(positions):
                source = (calls[k]["p"][0] == position - 1).nonzero().flatten()
                dest = (calls[k + 1]["p"][0] == position).nonzero().flatten()
                assert source.numel() == dest.numel() == 1
                state = calls[k]["h"][0, source.item()]
                errors.extend([(calls[k + 1]["x"][0, dest.item()] - state).abs().max().item(),
                               (vectors[k] - state).abs().max().item()])
        return vectors, max(errors, default=0.0), ids.shape[1]

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    out = root / "review_runs" / ("node_readout_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    print("OUTPUT DIRECTORY:", out, flush=True)
    embeddings = model.embedding.weight.detach().float().cpu()
    node_rows, group_rows, traces, prompts = [], [], [], []
    max_feedback_error = max_prefix_error = 0.0

    for order, index in enumerate(indices):
        item = data[index]
        rng = random.Random(args.seed + 10000 + index)
        edges = list(map(tuple, item["edges"]))
        rng.shuffle(edges)  # Same frozen prompt for every step of this graph.
        candidates = [item["target"], item["neg_target"]]
        if order % 2:
            candidates.reverse()  # Balanced answer position in the pilot.
        facts = " | ".join(f"{u} {v}" for u, v in edges)
        prompt = f"<eos> {facts} [Q] {candidates[0]} {candidates[1]} [R] {item['root']}"
        vectors, feedback_error, length = extract(prompt, args.steps, check=order < 2)
        max_feedback_error = max(max_feedback_error, feedback_error)
        # Verify that extracting all steps at once agrees with prefix-only runs.
        if order < 2:
            for n in range(1, args.steps):
                prefix, _, _ = extract(prompt, n)
                max_prefix_error = max(max_prefix_error,
                                       (prefix - vectors[:n]).abs().max().item())
        nodes, _, _, _ = graph_info(item)
        node_ids = sorted(nodes)
        token_ids = []
        for node in node_ids:
            tokens = tokenizer.encode(str(node), add_special_tokens=False)
            assert len(tokens) == 1
            token_ids.append(tokens[0])
        node_embeddings = embeddings[token_ids]
        dot = vectors @ node_embeddings.T
        norms = vectors.norm(dim=-1)
        node_norms = node_embeddings.norm(dim=-1)
        cosine = dot / (norms[:, None] * node_norms[None, :]).clamp_min(1e-12)

        for step in range(1, args.steps + 1):
            groups = node_groups(item, step)
            for col, node in enumerate(node_ids):
                row = {"graph_index": index, "step": step, "node": node,
                       "dot_product": dot[step-1, col].item(),
                       "cosine": cosine[step-1, col].item(),
                       "state_norm": norms[step-1].item(),
                       "node_embedding_norm": node_norms[col].item()}
                row.update({group: int(node in groups[group]) for group in GROUPS})
                node_rows.append(row)
            for group in GROUPS:
                cols = [col for col, node in enumerate(node_ids) if node in groups[group]]
                for metric, scores in (("dot_product", dot), ("cosine", cosine)):
                    group_rows.append({"graph_index": index, "step": step,
                                       "group": group, "metric": metric, "nodes": len(cols),
                                       "mean": scores[step-1, cols].mean().item() if cols else None})
        traces.append({"graph_index": index, "vectors": vectors})
        prompts.append({"graph_index": index, "prompt": prompt, "input_tokens": length,
                        "candidate_order": candidates, "edges": edges})
        if (order + 1) % 10 == 0 or order + 1 == len(indices):
            print(f"Analyzed {order + 1}/{len(indices)} graphs", flush=True)

    assert max_feedback_error <= 1e-6, max_feedback_error
    assert max_prefix_error <= 1e-5, max_prefix_error
    write_csv(out / "node_scores.csv", node_rows)
    write_csv(out / "graph_group_means.csv", group_rows)
    torch.save(traces, out / "latent_states.pt")
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2))

    def estimate(values):
        if not values:
            return {"graphs": 0, "mean": None, "ci_low": None, "ci_high": None}
        v = torch.tensor(values, dtype=torch.float64)
        generator = torch.Generator().manual_seed(args.seed)
        sample = torch.randint(len(v), (args.bootstrap, len(v)), generator=generator)
        bootstrap_means = v[sample].mean(dim=1)
        return {"graphs": len(values), "mean": v.mean().item(),
                "ci_low": torch.quantile(bootstrap_means, 0.025).item(),
                "ci_high": torch.quantile(bootstrap_means, 0.975).item()}

    summary_rows, contrasts = [], []
    for step in range(1, args.steps + 1):
        for metric in ("dot_product", "cosine"):
            cells = {}
            for group in GROUPS:
                cells[group] = {r["graph_index"]: r["mean"] for r in group_rows
                                if r["step"] == step and r["metric"] == metric
                                and r["group"] == group and r["mean"] is not None}
                summary_rows.append({"step": step, "metric": metric, "group": group,
                                     **estimate(list(cells[group].values()))})
            for positive in ("reachable_up_to_k", "walk_frontier", "reference_optimal"):
                negative = "not_reached_up_to_k"
                paired = sorted(cells[positive].keys() & cells[negative].keys())
                contrast = {"step": step, "metric": metric,
                            "contrast": positive + " minus " + negative,
                            **estimate([cells[positive][i] - cells[negative][i] for i in paired])}
                contrasts.append(contrast)
                if metric == "dot_product" and positive == "walk_frontier":
                    print("FRONTIER CONTRAST:", json.dumps(contrast), flush=True)
    write_csv(out / "group_summary.csv", summary_rows)
    write_csv(out / "paired_contrasts.csv", contrasts)

    plot_status = "not attempted"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        plot_status = "unavailable: " + str(error)
    else:
        plt.rcParams.update({"font.family": "serif", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        palette = {"not_reached_up_to_k": ("#A64B38", "Not reached within k"),
                   "reachable_up_to_k": ("#356891", "Reached within k"),
                   "walk_frontier": ("#398061", "Exact-k walk frontier"),
                   "reference_optimal": ("#7A5C94", "Reference optimal")}
        fig, axes = plt.subplots(1, args.steps, figsize=(3.5 * args.steps, 3.2), squeeze=False)
        for step in range(1, args.steps + 1):
            ax = axes[0, step-1]
            selected = [r for r in node_rows if r["step"] == step]
            lo, hi = min(r["dot_product"] for r in selected), max(r["dot_product"] for r in selected)
            for group, (color, label) in palette.items():
                values = [r["dot_product"] for r in selected if r[group]]
                if values:
                    ax.hist(values, bins=25, range=(lo, hi), density=True,
                            histtype="step", linewidth=1.4, color=color, label=label)
            ax.set_title(f"Latent step {step}")
            ax.set_xlabel("State-node embedding dot product")
            if step == 1:
                ax.set_ylabel("Density")
        axes[0, -1].legend(fontsize=7)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(out / ("node_score_distributions." + suffix), dpi=250)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4))
        for group, (color, label) in palette.items():
            rows = sorted([r for r in summary_rows if r["metric"] == "dot_product"
                           and r["group"] == group and r["graphs"]], key=lambda r: r["step"])
            means = [r["mean"] for r in rows]
            ax.errorbar([r["step"] for r in rows], means,
                        yerr=[[r["mean"] - r["ci_low"] for r in rows],
                              [r["ci_high"] - r["mean"] for r in rows]],
                        marker="o", markersize=4, linewidth=1.4, capsize=3, color=color, label=label)
        ax.set_xticks(list(range(1, args.steps + 1)))
        ax.set_xlabel("Latent injection step")
        ax.set_ylabel("Mean dot product (equal weight per graph)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(out / ("graph_mean_scores." + suffix), dpi=250)
        plt.close(fig)
        plot_status = "saved PNG and SVG"

    summary = {
        "analysis": "Fixed-cohort adaptation of RBS notebook node-embedding readout",
        "checkpoint": str(checkpoint), "checkpoint_sha256": expected_hash,
        "checkpoint_variant": "published example; not exact reported-accuracy reproduction",
        "dataset": str(dataset), "dataset_sha256": sha256(dataset),
        "dataset_records": len(data), "eligible_fixed_cohort": len(pool),
        "graphs": len(indices), "graph_indices": indices, "steps": args.steps,
        "seed": args.seed, "precision": str(next(base.parameters()).dtype),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "attention_backend": "eager",
        "strict_loading": True, "maximum_feedback_error": max_feedback_error,
        "maximum_prefix_extraction_error": max_prefix_error,
        "extraction_check_graphs": min(2, len(indices)),
        "correctness_filter": False, "answer_generation": False,
        "bootstrap_replicates": args.bootstrap, "bootstrap_unit": "graph",
        "confidence_intervals": "pointwise 95% percentile; not simultaneous",
        "pooling": "histograms pool graph-node pairs; estimates average graph means",
        "definitions": {
            "walk_frontier": "Nodes reachable by a walk of exactly k edges; may overlap earlier frontiers",
            "reachable_up_to_k": "Union of walk frontiers from 0 through k",
            "not_reached_up_to_k": "Present graph nodes outside reachable_up_to_k; can include later-reachable nodes",
            "reference_optimal": "Dataset neighbor_k, solver-verified nodes at depth k on shortest root-to-target paths",
        },
        "limitations": ["Restricted four-step-annotated cohort when steps=4",
                        "Nested node categories are not independent groups",
                        "Embedding readout is not proof of causal use or BFS",
                        "The mapping from latent step to graph depth follows the notebook and must be tested later",
                        "Cosine and bootstrap analysis are additions to the notebook"],
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in
                          [root / "coconut.py", root / "stokenizer.py", config_path]},
        "analysis_script_sha256": sha256(__file__), "plots": plot_status,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("PILOT COMPLETE:", out, flush=True)


if __name__ == "__main__":
    main()
