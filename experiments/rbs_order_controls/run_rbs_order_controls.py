#!/usr/bin/env python3
"""Paired order controls of the frozen RBS cohort; no training or generation.
Keep run_rbs_node_readout.py beside this file. Only the first --graphs parent
records are used. Original prompts are replayed, never resampled.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

DATA_SHA = '584083aebf7ee583308d56a909d7e62bd560ad9acd0bf0a1dec514bff9143267'
CKPT_SHA = '6fafa619ab8b57c065e9f3fe94d3eecb13a42d880441d582b1b0bfa3ae8a6fc0'
HELPER_SHA = '65c4b0f5403b834134e1ac4f6569646f1fc9e0983815ed751d7a722c6d983aa3'
CONDITIONS = ('original', 'candidate_swap', 'edge_permutation')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def variants(row, seed):
    prompt = row['prompt']
    match = re.fullmatch(r'<eos> (.+) \[Q\] (\d+) (\d+) \[R\] (\d+)', prompt)
    assert match, 'Unexpected native prompt format'
    facts, a, b, root = match.groups()
    edges = [tuple(map(int, edge.split())) for edge in facts.split(' | ')]
    assert all(len(edge) == 2 for edge in edges)
    assert edges == list(map(tuple, row['edges']))
    assert [int(a), int(b)] == row['candidate_order'] and a != b
    shuffled = edges.copy()
    random.Random(seed + 50000 + row['graph_index']).shuffle(shuffled)
    if shuffled == edges:
        shuffled = edges[1:] + edges[:1]
    assert shuffled != edges and sorted(shuffled) == sorted(edges)
    changed_facts = ' | '.join(f'{u} {v}' for u, v in shuffled)
    result = {
        'original': prompt,
        'candidate_swap': f'<eos> {facts} [Q] {b} {a} [R] {root}',
        'edge_permutation': f'<eos> {changed_facts} [Q] {a} {b} [R] {root}',
    }
    lengths = {len(p.split()) for p in result.values()}
    assert len(lengths) == 1
    assert sum(x != y for x, y in zip(prompt.split(), result['candidate_swap'].split())) == 2
    return result


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--parent-dir', type=Path,
                        default=Path('review_runs/node_readout_20261017_161848_557835'))
    parser.add_argument('--graphs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--audit-parent-zip', type=Path,
                        help='CPU-only prompt audit; no model imports')
    args = parser.parse_args()
    assert args.graphs > 0 and args.bootstrap >= 100
    if args.audit_parent_zip:
        with ZipFile(args.audit_parent_zip) as z:
            prompts = json.loads(z.read('prompts.json'))
        for row in prompts:
            variants(row, args.seed)
        print(json.dumps({'prompt_audit_passed': True, 'graphs': len(prompts),
                          'conditions': list(CONDITIONS), 'model_run': False}, indent=2))
        return

    root = args.repo_root.resolve()
    parent = args.parent_dir
    if not parent.is_absolute():
        parent = root / parent
    summary = json.loads((parent / 'summary.json').read_text())
    prompts = json.loads((parent / 'prompts.json').read_text())
    assert summary['dataset_sha256'] == DATA_SHA
    assert summary['checkpoint_sha256'] == CKPT_SHA
    assert summary['steps'] == 4 and args.graphs <= len(prompts)
    assert [p['graph_index'] for p in prompts] == summary['graph_indices']
    prompts = prompts[:args.graphs]
    for row in prompts:
        variants(row, args.seed)
    helper = root / 'run_rbs_node_readout.py'
    assert digest(helper) == HELPER_SHA, 'Unexpected helper version'
    sys.path.insert(0, str(root))
    from run_rbs_node_readout import audit, node_groups, graph_info
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from coconut import Coconut
    from stokenizer import STokenizer
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = root / 'data/prosqa_test_graph_4_coconut.json'
    checkpoint = root / 'artifacts/coconut_theory/checkpoint_300'
    assert digest(dataset) == DATA_SHA and digest(checkpoint) == CKPT_SHA
    data = json.loads(dataset.read_text())
    eligible = set(audit(data, 4))
    for row in prompts:
        index = row['graph_index']
        assert index in eligible
        item = data[index]
        assert sorted(map(tuple, row['edges'])) == sorted(map(tuple, item['edges']))
        assert set(row['candidate_order']) == {item['target'], item['neg_target']}
        assert row['prompt'].endswith(f"[R] {item['root']}")
        graph_info(item)
    tokenizer = STokenizer()
    config_path = root / 'configs/symbol-2layer-8head-768dim.json'
    config = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
    base = AutoModelForCausalLM.from_config(config, attn_implementation='eager').float()
    latent, start, end, eos = [tokenizer.convert_tokens_to_ids(s) for s in
        ('<|latent|>', '<|start-latent|>', '<|end-latent|>', '<eos>')]
    assert config.vocab_size == len(tokenizer) == 40
    model = Coconut(base, latent, start, end, eos)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to('cuda')
    model.eval()
    print('STRICT LOADING AND FROZEN COHORT AUDIT PASSED', flush=True)

    def extract(prompt, check):
        full = prompt + ' ' + ' '.join(['<|latent|>'] * 4)
        ids = torch.tensor([tokenizer.encode(full, add_special_tokens=False)], device='cuda')
        assert ids.shape[1] <= config.n_positions
        positions = (ids[0] == latent).nonzero().flatten().tolist()
        assert len(positions) == 4
        calls, handles = [], []
        if check:
            def before(module, pargs, kwargs):
                calls.append({'p': kwargs['position_ids'].detach().cpu(),
                              'x': kwargs['inputs_embeds'].detach().cpu()})
            def after(module, pargs, kwargs, output):
                calls[-1]['h'] = output.hidden_states[-1].detach().cpu()
            handles = [base.register_forward_pre_hook(before, with_kwargs=True),
                       base.register_forward_hook(after, with_kwargs=True)]
        try:
            with torch.no_grad():
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    labels=ids.clone(), position_ids=torch.arange(ids.shape[1], device='cuda').unsqueeze(0))
        finally:
            for handle in handles:
                handle.remove()
        assert torch.isfinite(output.logits).all()
        vectors = output.inputs_embeds[0, positions, :].detach().float().cpu()
        assert torch.isfinite(vectors).all()
        error = 0.0
        if check:
            assert len(calls) == 5
            for k, position in enumerate(positions):
                src = (calls[k]['p'][0] == position - 1).nonzero().flatten()
                dst = (calls[k+1]['p'][0] == position).nonzero().flatten()
                assert src.numel() == dst.numel() == 1
                state = calls[k]['h'][0, src.item()]
                error = max(error, (vectors[k] - state).abs().max().item(),
                    (calls[k+1]['x'][0, dst.item()] - state).abs().max().item())
        assert error <= 1e-6, 'Feedback extraction failed'
        return vectors, error, ids[0].detach().cpu().tolist()

    with (parent / 'node_scores.csv').open() as f:
        old = {(int(r['graph_index']), int(r['step']), int(r['node'])): r
               for r in csv.DictReader(f)}
    out = root / 'review_runs' / ('order_controls_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT DIRECTORY:', out, flush=True)
    embeddings = model.embedding.weight.detach().float().cpu()
    node_rows, contrast_rows, state_rows, prompt_rows, traces = [], [], [], [], []
    max_replay_error = max_feedback_error = repeatability_error = 0.0
    for order, row in enumerate(prompts):
        index = row['graph_index']
        item = data[index]
        nodes = sorted(graph_info(item)[0])
        node_token_ids = [tokenizer.encode(str(node), add_special_tokens=False) for node in nodes]
        assert all(len(ids) == 1 for ids in node_token_ids)
        emb = embeddings[[ids[0] for ids in node_token_ids]]
        states = {}
        token_ids = {}
        for condition, prompt in variants(row, args.seed).items():
            vectors, error, ids = extract(prompt, order < 2)
            max_feedback_error = max(max_feedback_error, error)
            states[condition] = vectors
            token_ids[condition] = ids
            assert len(ids) == row['input_tokens'], 'Prompt length changed'
            if order == 0 and condition == 'original':
                repeated, _, _ = extract(prompt, False)
                repeatability_error = (vectors - repeated).abs().max().item()
                assert repeatability_error <= 1e-6
            prompt_rows.append({'graph_index': index, 'condition': condition,
                                'prompt': prompt, 'input_tokens': len(ids),
                                'changed_token_positions': sum(a != b for a, b in zip(ids, token_ids['original']))})
            all_dot = vectors @ emb.T
            all_cosine = all_dot / (vectors.norm(dim=-1)[:, None] * emb.norm(dim=-1)[None, :]).clamp_min(1e-12)
            for k, vector in enumerate(vectors, 1):
                dot = all_dot[k-1]
                cosine = all_cosine[k-1]
                groups = node_groups(item, k)
                for j, node in enumerate(nodes):
                    if condition == 'original':
                        previous = old[(index, k, node)]
                        max_replay_error = max(max_replay_error,
                            abs(float(dot[j]) - float(previous['dot_product'])),
                            abs(float(cosine[j]) - float(previous['cosine'])))
                    node_rows.append({'graph_index': index, 'condition': condition, 'step': k,
                                      'node': node, 'dot_product': dot[j].item(), 'cosine': cosine[j].item()})
                for metric, scores in [('dot_product', dot), ('cosine', cosine)]:
                    means = {}
                    for group in ('walk_frontier', 'not_reached_up_to_k'):
                        selected = [j for j, node in enumerate(nodes) if node in groups[group]]
                        assert selected
                        means[group] = scores[selected].mean().item()
                    contrast_rows.append({'graph_index': index, 'condition': condition,
                        'step': k, 'metric': metric,
                        'contrast': means['walk_frontier'] - means['not_reached_up_to_k']})
        assert sum(a != b for a, b in zip(token_ids['original'], token_ids['candidate_swap'])) == 2
        assert max_replay_error <= 1e-4, 'Original node scores did not replay: inspect before interpreting controls'
        for condition in CONDITIONS[1:]:
            for k, (original, changed) in enumerate(zip(states['original'], states[condition]), 1):
                state_rows.append({'graph_index': index, 'condition': condition, 'step': k,
                    'cosine_similarity': torch.nn.functional.cosine_similarity(original, changed, dim=0).item(),
                    'normalized_displacement': ((changed-original).norm()/original.norm().clamp_min(1e-12)).item()})
        traces.append({'graph_index': index, 'states': states})
        print(f'Analyzed {order+1}/{len(prompts)} graphs', flush=True)

    def estimate(values, rng):
        values = torch.tensor(values, dtype=torch.float64)
        draws = torch.randint(len(values), (args.bootstrap, len(values)), generator=rng)
        sampled = values[draws].mean(dim=1)
        return {'graphs': len(values), 'mean': values.mean().item(),
                'ci_low': torch.quantile(sampled, .025).item(),
                'ci_high': torch.quantile(sampled, .975).item()}
    rng = torch.Generator().manual_seed(args.seed)
    aggregates = []
    lookup = {(r['graph_index'], r['condition'], r['step'], r['metric']): r['contrast']
              for r in contrast_rows}
    indices = [r['graph_index'] for r in prompts]
    for condition in CONDITIONS:
        for step in range(1, 5):
            for metric in ('dot_product', 'cosine'):
                values = [lookup[i, condition, step, metric] for i in indices]
                aggregates.append({'condition': condition, 'step': step, 'metric': metric,
                    'estimate': 'frontier_minus_not_reached', **estimate(values, rng)})
                if condition != 'original':
                    delta = [lookup[i, condition, step, metric] - lookup[i, 'original', step, metric] for i in indices]
                    aggregates.append({'condition': condition, 'step': step, 'metric': metric,
                        'estimate': 'paired_change_from_original', **estimate(delta, rng)})
    result = {
        'analysis': 'Paired candidate-order and edge-order controls of frozen RBS cohort',
        'graphs': len(prompts), 'graph_indices': indices, 'conditions': list(CONDITIONS),
        'steps': 4, 'parent_directory': str(parent), 'parent_summary_sha256': digest(parent/'summary.json'),
        'dataset_sha256': DATA_SHA, 'checkpoint_sha256': CKPT_SHA,
        'maximum_original_node_score_replay_error': max_replay_error,
        'maximum_feedback_error': max_feedback_error, 'repeatability_error': repeatability_error,
        'feedback_check_graphs_per_condition': min(2, len(prompts)),
        'bootstrap_replicates': args.bootstrap, 'bootstrap_unit': 'graph; paired changes within graph',
        'confidence_intervals': 'pointwise 95% percentile; not simultaneous',
        'seed': args.seed, 'precision': 'float32', 'attention_backend': 'eager',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'correctness_filter': False, 'answer_generation': False, 'training': False,
        'analysis_script_sha256': digest(__file__),
        'limitations': ['One published visualization checkpoint; no accuracy reproduction.',
            'One edge permutation per graph; selected four-step cohort.',
            'Order sensitivity is not evidence of reasoning failure or causal use.',
            'An interval containing zero does not establish equivalence.',
            'Candidate swaps and edge permutations change different numbers of tokens.']}
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    (out/'prompts.json').write_text(json.dumps(prompt_rows, indent=2))
    for name, rows in [('node_scores.csv', node_rows), ('graph_contrasts.csv', contrast_rows),
                       ('state_changes.csv', state_rows), ('paired_estimates.csv', aggregates)]:
        write_csv(out/name, rows)
    torch.save(traces, out/'latent_states.pt')
    archive = out/'order_control_tables.zip'
    with ZipFile(archive, 'x', ZIP_DEFLATED) as z:
        for name in ('summary.json', 'prompts.json', 'node_scores.csv', 'graph_contrasts.csv',
                     'state_changes.csv', 'paired_estimates.csv'):
            z.write(out/name, arcname=name)
    print(json.dumps(result, indent=2))
    print('UPLOAD TABLES ZIP:', archive, flush=True)


if __name__ == '__main__':
    main()
