"""Read-only diagnostics for run_rq2_workspace_feasibility_v1 checkpoints.

Strictly reloads each saved model; reproduces original development predictions;
profiles ordinary inference; and performs same-shape workspace interventions.
No training, no new checkpoint and no automatic hybrid/architecture claim.
"""
import argparse
import gc
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_data(train, development):
    import re
    signatures = []
    for rows in (train, development):
        groups = defaultdict(list)
        sigs = set()
        for row in rows:
            edges = list(map(tuple, row['edges']))
            assert len(dict(edges)) == len(edges)
            assert sorted(re.findall(r'([A-Z]) -> ([A-Z])\.', row['question'])) == sorted(edges)
            query = re.search(r'Start at ([A-Z])\. Follow exactly (\d+) directed edges', row['question'])
            assert query and query.group(1) == row['start'] and int(query.group(2)) == row['hops']
            current = row['start']
            successor = dict(edges)
            for _ in range(row['hops']):
                current = successor[current]
            assert current == row['answer']
            groups[row['base_id']].append(row)
            sigs.add(tuple(sorted(edges)))
        for pair in groups.values():
            assert len(pair) == 2 and {r['query_index'] for r in pair} == {0, 1}
            assert pair[0]['answer'] != pair[1]['answer']
            assert sorted(map(tuple, pair[0]['edges'])) == sorted(map(tuple, pair[1]['edges']))
        signatures.append(sigs)
    assert not signatures[0] & signatures[1]
    return {'training_records': len(train), 'development_records': len(development),
            'training_answer_labels': len({r['answer'] for r in train}),
            'solver_and_prompt_audit': True, 'identical_labelled_edge_sets_overlap': 0,
            'structural_isomorphism_disjoint': False}


def interval(values, replicates, rng):
    import numpy as np
    a = np.asarray(values, dtype=float)
    assert a.size > 0 and np.isfinite(a).all()
    means = a[rng.integers(0, len(a), size=(replicates, len(a)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {'mean': float(a.mean()), 'ci_low': float(low), 'ci_high': float(high), 'base_groups': len(a)}


def main():
    import numpy as np
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('../gpt2_model'))
    parser.add_argument('--groups', type=int, default=2, help='Use 0 for all development groups.')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=732026)
    args = parser.parse_args()
    assert args.groups >= 0 and args.warmup >= 1 and args.repeats >= 2 and args.bootstrap >= 100
    assert torch.cuda.is_available(), 'Run on a CUDA node.'
    assert torch.__version__.split('+')[0] == '2.4.1', 'Reproduce the original environment first.'
    assert transformers.__version__ == '4.46.2', 'Reproduce the original environment first.'
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = args.run_dir.resolve()
    load_json = lambda p: json.loads(p.read_text())
    cfg = load_json(run / 'configuration.json')
    assert cfg['arguments']['save_weights'], 'This run did not request saved weights.'
    assert cfg['cache'] is False and cfg['dropout'] == 0.0
    for filename, expected in cfg['data_sha256'].items():
        assert sha_file(run / filename) == expected, ('Dataset hash mismatch', filename)
    train = load_json(run / 'train.json')
    dev_all = load_json(run / 'development.json')
    data_audit = audit_data(train, dev_all)
    base_ids = list(dict.fromkeys(x['base_id'] for x in dev_all))
    if args.groups:
        assert args.groups <= len(base_ids)
        base_ids = base_ids[:args.groups]
    chosen = set(base_ids)
    dev = [x for x in dev_all if x['base_id'] in chosen]
    out = run / ('diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir.resolve()), local_files_only=True)
    prefix_ids = tokenizer.encode('###', add_special_tokens=False)
    assert len(prefix_ids) == 1 and tokenizer.eos_token_id is not None
    encoded = {}
    by_pair = defaultdict(list)
    for row in dev:
        key = (row['base_id'], row['query_index'])
        q = torch.tensor([tokenizer.encode(row['question'] + '\n', add_special_tokens=False)], device=device)
        target = tokenizer.encode(row['answer'], add_special_tokens=False)
        assert len(target) == 1
        encoded[key] = (q, target[0], row)
        by_pair[row['base_id']].append(key)
    opposite = {}
    for keys in by_pair.values():
        assert len(keys) == 2
        opposite[keys[0]], opposite[keys[1]] = keys[1], keys[0]

    class WorkspaceModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def pass_states(self, embeddings):
            length = embeddings.shape[1]
            assert length <= self.base.config.n_positions
            return self.base.transformer(
                inputs_embeds=embeddings,
                position_ids=torch.arange(length, device=device).unsqueeze(0),
                attention_mask=torch.ones((1, length), device=device, dtype=torch.long),
                use_cache=False, output_hidden_states=False, return_dict=True,
            ).last_hidden_state

        def build_workspace(self, question, strategy, steps):
            context = self.base.transformer.wte(question)
            qlen = context.shape[1]
            workspace = self.base.transformer.wte(torch.tensor([[tokenizer.eos_token_id]], device=device))
            lengths = []
            for _ in range(steps):
                old = workspace
                inp = torch.cat([context, old], dim=1)
                lengths.append(inp.shape[1])
                states = self.pass_states(inp)
                updates = states[:, qlen:, :]
                workspace = torch.cat([old, updates[:, -1:, :]], dim=1) if strategy == 'EXPAND' else updates
            return context, workspace, lengths

        def readout(self, context, workspace):
            prefix = self.base.transformer.wte(torch.tensor([prefix_ids], device=device))
            inp = torch.cat([context, workspace, prefix], dim=1)
            return self.base.lm_head(self.pass_states(inp)[:, -1, :])

        def forward(self, question, strategy, steps):
            context, workspace, _ = self.build_workspace(question, strategy, steps)
            return self.readout(context, workspace)

    # The original checkpoint metadata contains PyTorch's own TorchVersion string
    # subclass. Allow exactly that known class while retaining weights_only=True.
    if type(torch.__version__) is not str:
        torch.serialization.add_safe_globals([type(torch.__version__)])

    conditions = ['normal', 'self_replay', 'opposite_query_workspace', 'norm_matched_random', 'zero_workspace']
    info = {'source_run': str(run), 'source_configuration_sha256': sha_file(run / 'configuration.json'),
            'diagnostic_script_sha256': sha_file(Path(__file__)),
            'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'gpu': torch.cuda.get_device_name(0), 'precision': 'FP32; TF32 disabled',
            'base_groups': len(base_ids), 'records': len(dev), 'base_ids': base_ids,
            'data_audit': data_audit, 'conditions': conditions,
            'bootstrap': {'replicates': args.bootstrap, 'unit': 'base graph with both queries',
                          'intervals': 'pointwise percentile 95%; exploratory, no multiplicity correction'},
            'compute_convention': 'Two FLOPs per multiply-add. Profiler estimates and analytical cross-check count dense matmuls only; elementwise ops, softmax, norms, embeddings and tensor movement are excluded.',
            'limitations': ['One training seed; simple two-chain development task.',
                            'Native schedules are profiled, not automatically matched or optimized.',
                            'Readout sees question context; interventions test conditional workspace dependence.',
                            'Zero/random interventions may cause general damage; query-donor effects require controls.',
                            'Same-shape interventions preserve recipient readout positions, not necessarily the latent distribution.',
                            'Direct reference retains the initial EOS workspace; its perturbations are a control.']}
    dump('configuration.json', info)
    print('OUTPUT DIRECTORY:', out.resolve(), flush=True)
    results = {}
    torch.set_grad_enabled(False)
    for strategy in ('DIRECT', 'EXPAND', 'REFINE'):
        steps = 0 if strategy == 'DIRECT' else cfg['arguments']['steps']
        checkpoint = run / strategy / 'final_weights.pt'
        assert checkpoint.is_file(), checkpoint
        base = AutoModelForCausalLM.from_pretrained(str(args.model_dir.resolve()), local_files_only=True,
                                                   attn_implementation='eager', torch_dtype=torch.float32)
        for module in base.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        model = WorkspaceModel(base)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['strategy'] == strategy and saved['steps'] == steps
        assert saved['configuration']['initial_state_sha256'] == cfg['initial_state_sha256']
        model.load_state_dict(saved['model_state_dict'], strict=True)
        del saved
        model.to(device).eval()
        assert all(type(block.attn).__name__ == 'GPT2Attention' for block in model.base.transformer.h)
        expected = {(x['base_id'], x['query_index']): x for x in load_json(run / strategy / 'development.json')['results']}
        workspaces, normal_predictions, normal_logits, lengths, norms = {}, {}, {}, {}, {}
        # Compute each original workspace once for interventions; full forward is
        # used separately for inference profiling and repeatability checks.
        for key, (q, target, row) in encoded.items():
            context, workspace, passes = model.build_workspace(q, strategy, steps)
            logits = model.readout(context, workspace)
            pred = int(logits.argmax(dim=-1))
            assert pred == expected[key]['prediction_token_id'], ('Original prediction replay mismatch', strategy, key)
            workspaces[key] = workspace.detach().clone()
            normal_predictions[key] = pred
            normal_logits[key] = logits.detach().clone()
            lengths[key] = passes + [q.shape[1] + workspace.shape[1] + len(prefix_ids)]
            norms[key] = workspace.norm(dim=-1).tolist()[0]
        first = next(iter(encoded))
        qa = encoded[first][0]
        reconstructed = model(qa, strategy, steps)
        reconstruction_error = float((reconstructed - normal_logits[first]).abs().max())
        assert reconstruction_error <= 1e-6, reconstruction_error
        del reconstructed

        rows = []
        for group_index, base_id in enumerate(base_ids):
            for key in by_pair[base_id]:
                q, target, item = encoded[key]
                other_key = opposite[key]
                other_target = encoded[other_key][1]
                w = workspaces[key]
                context = model.base.transformer.wte(q)
                generator = torch.Generator(device=device).manual_seed(args.seed + group_index)
                noise = torch.randn(w.shape, generator=generator, device=device, dtype=w.dtype)
                noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12) * w.norm(dim=-1, keepdim=True)
                donor_length_matches = q.shape[1] == encoded[other_key][0].shape[1]
                options = {'normal': w, 'self_replay': w.clone(),
                           'opposite_query_workspace': workspaces[other_key] if donor_length_matches else None,
                           'norm_matched_random': noise, 'zero_workspace': torch.zeros_like(w)}
                for condition, replacement in options.items():
                    if replacement is None:
                        continue
                    assert replacement.shape == w.shape
                    logits = normal_logits[key] if condition == 'normal' else model.readout(context, replacement)
                    if condition == 'self_replay':
                        assert float((logits-normal_logits[key]).abs().max()) <= 1e-6
                    predicted = int(logits.argmax(dim=-1))
                    rows.append({'base_id': base_id, 'query_index': item['query_index'],
                                 'condition': condition, 'target': item['answer'],
                                 'other_target': encoded[other_key][2]['answer'],
                                 'prediction': tokenizer.decode([predicted]), 'prediction_token_id': predicted,
                                 'correct': predicted == target, 'predicts_other_target': predicted == other_target,
                                 'prediction_changed': predicted != normal_predictions[key],
                                 'target_minus_other_logit_margin': float(logits[0, target] - logits[0, other_target]),
                                 'normal_target_minus_other_logit_margin': float(normal_logits[key][0, target] - normal_logits[key][0, other_target]),
                                 'donor_prompt_length_matched': donor_length_matches,
                                 'workspace_length': w.shape[1]})
        # Report group-paired changes, averaging both queries before resampling.
        summaries, contrasts = {}, {}
        normal_map = {(x['base_id'], x['query_index']): x for x in rows if x['condition'] == 'normal'}
        for condition in conditions:
            selected = [x for x in rows if x['condition'] == condition]
            if not selected:
                summaries[condition] = {'records': 0, 'reason': 'No length-matched paired donors'}
                continue
            grouped = defaultdict(list)
            for row in selected:
                grouped[row['base_id']].append(row)
            assert all(len(pair) == 2 for pair in grouped.values())
            summaries[condition] = {'records': len(selected), 'groups': len(grouped),
                                    'accuracy': sum(x['correct'] for x in selected) / len(selected),
                                    'both_queries_correct': sum(all(x['correct'] for x in pair) for pair in grouped.values()),
                                    'predicts_other_target_fraction': sum(x['predicts_other_target'] for x in selected) / len(selected),
                                    'prediction_changed_fraction': sum(x['prediction_changed'] for x in selected) / len(selected)}
            if condition != 'normal':
                accuracy_delta, margin_delta, other_delta = [], [], []
                for pair in grouped.values():
                    reference = [normal_map[(x['base_id'], x['query_index'])] for x in pair]
                    accuracy_delta.append(sum(int(x['correct'])-int(y['correct']) for x,y in zip(pair,reference))/2)
                    margin_delta.append(sum(x['target_minus_other_logit_margin']-y['target_minus_other_logit_margin'] for x,y in zip(pair,reference))/2)
                    other_delta.append(sum(int(x['predicts_other_target'])-int(y['predicts_other_target']) for x,y in zip(pair,reference))/2)
                rng = np.random.default_rng(args.seed)
                contrasts[condition] = {'accuracy_change_vs_normal': interval(accuracy_delta, args.bootstrap, rng),
                                        'target_minus_other_margin_change': interval(margin_delta, args.bootstrap, rng),
                                        'other_target_prediction_change': interval(other_delta, args.bootstrap, rng)}
        dump(strategy + '_intervention_predictions.json', rows)
        dump(strategy + '_intervention_summary.json', {'conditions': summaries, 'paired_contrasts': contrasts})

        # Release cached intervention tensors before measuring ordinary inference.
        del context, workspace, w, replacement, logits, noise, options, workspaces, normal_logits
        # Profile one representative of every input length, with observed bucket counts.
        buckets = defaultdict(list)
        for key, (q, _, _) in encoded.items():
            buckets[q.shape[1]].append(key)
        profile_rows = []
        for qlen, keys in sorted(buckets.items()):
            q = encoded[keys[0]][0]
            for _ in range(args.warmup):
                model(q, strategy, steps)
            torch.cuda.synchronize()
            baseline_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(q, strategy, steps)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            peak = torch.cuda.max_memory_allocated()
            # CPU operator events carry shapes and matmul FLOP estimates even
            # when their tensor operations execute on CUDA. Latency is measured
            # independently above, outside the profiler.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True, with_flops=True) as profiler:
                model(q, strategy, steps)
                torch.cuda.synchronize()
            counted = [{'operator': e.key, 'calls': e.count, 'flops': int(e.flops)}
                       for e in profiler.key_averages() if e.flops and e.flops > 0]
            matmul_ops = {'aten::mm', 'aten::bmm', 'aten::addmm', 'aten::baddbmm'}
            profiler_matmul = sum(e['flops'] for e in counted if e['operator'] in matmul_ops)
            assert profiler_matmul > 0, 'Profiler did not count matmul operations.'
            pass_lengths = lengths[keys[0]]
            d = model.base.config.n_embd
            layers = model.base.config.n_layer
            analytical = layers * sum(24*length*d*d + 4*length*length*d for length in pass_lengths)
            analytical += 2*d*model.base.config.vocab_size  # Last-position LM head only.
            relative_error = abs(profiler_matmul-analytical) / analytical
            profile_rows.append({'question_tokens': qlen, 'records_in_bucket': len(keys),
                                 'pass_lengths_including_final_readout': pass_lengths,
                                 'profiler_matmul_flops': profiler_matmul,
                                 'analytical_dense_matmul_flops': analytical,
                                 'profiler_vs_analytical_relative_error': relative_error,
                                 'crosscheck_within_1_percent': relative_error <= 0.01,
                                 'all_positive_profiler_operator_counts': counted,
                                 'latency_gpu_ms_median': float(np.median(timings)),
                                 'latency_gpu_ms_min': min(timings), 'latency_gpu_ms_max': max(timings),
                                 'latency_repeats': args.repeats, 'latency_warmup': args.warmup,
                                 'memory_baseline_allocated_gib': baseline_memory / 1024**3,
                                 'memory_peak_allocated_gib': peak / 1024**3,
                                 'memory_incremental_peak_gib': (peak-baseline_memory) / 1024**3})
        dump(strategy + '_inference_profile.json', profile_rows)
        weighted_flops = sum(x['profiler_matmul_flops']*x['records_in_bucket'] for x in profile_rows)/len(dev)
        results[strategy] = {'checkpoint_sha256': sha_file(checkpoint), 'strict_load': True,
                             'original_prediction_replay_matches': len(dev),
                             'reconstruction_max_logit_error_checked_record': reconstruction_error,
                             'normal': summaries['normal'], 'interventions': summaries,
                             'mean_profiled_matmul_flops': weighted_flops,
                             'profile_crosschecks_all_within_1_percent': all(x['crosscheck_within_1_percent'] for x in profile_rows),
                             'length_bucket_count': len(profile_rows)}
        dump('summary.json', results)
        print('DIAGNOSTIC', strategy, json.dumps(results[strategy]), flush=True)
        # Drop all references to this model's computation before the next load.
        del model, base, module
        gc.collect()
        torch.cuda.empty_cache()
    if results['EXPAND']['profile_crosschecks_all_within_1_percent'] and results['REFINE']['profile_crosschecks_all_within_1_percent']:
        e = results['EXPAND']['mean_profiled_matmul_flops']
        r = results['REFINE']['mean_profiled_matmul_flops']
        dump('cost_comparison.json', {'expand_over_refine_mean_matmul_flops': e/r,
                                     'relative_gap_against_lower_mean': abs(e-r)/min(e,r),
                                     'scope': 'Measured dense-matmul convention; omitted operations listed in configuration. Mean comparison alone does not enforce per-input budget ceilings.'})
    from zipfile import ZipFile, ZIP_DEFLATED
    archive = out / 'diagnostic_tables.zip'
    with ZipFile(archive, 'x', ZIP_DEFLATED) as z:
        for path in sorted(out.glob('*.json')):
            z.write(path, arcname=path.name)
    print('UPLOAD ZIP:', archive.resolve(), flush=True)
    print('No optimizer steps performed; checkpoints unchanged.', flush=True)


if __name__ == '__main__':
    main()
