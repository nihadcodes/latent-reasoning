"""Read-only execution audit of uploaded legacy RQ2 baselines. No optimizer step.
Run from mentor_experiment with its train_matched_logical.py and local GPT-2.
This audits the legacy implementation, not a corrected or compute-matched model.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    assert (root / 'train_matched_logical.py').is_file(), root
    sys.path.insert(0, str(root))
    import train_matched_logical as exp

    # The legacy loader resolves MODEL_ID relative to its process directory.
    exp.MODEL_ID = str(root.parent / 'gpt2_model')
    assert torch.cuda.is_available(), 'Run on an allocated GPU node.'
    exp.device = torch.device('cuda')
    exp.LATENT_STEPS = 2
    out = root / 'rq2_feasibility_audits' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    items = [
        {'base_id': 'audit_0000', 'condition': 'original', 'query_index': 0,
         'question': 'Directed edges: X -> B. P -> Q. B -> D. Q -> K.\nStart at X. Follow exactly 2 directed edges. Which node do you reach?',
         'answer': 'D'},
        {'base_id': 'audit_0000', 'condition': 'original', 'query_index': 1,
         'question': 'Directed edges: X -> B. P -> Q. B -> D. Q -> K.\nStart at P. Follow exactly 2 directed edges. Which node do you reach?',
         'answer': 'K'},
    ]
    report = {'purpose': 'Legacy execution audit; no training and no compute matching',
              'python': sys.version, 'torch': torch.__version__,
              'transformers': transformers.__version__,
              'gpu': torch.cuda.get_device_name(0), 'seed': 2026, 'steps': 2,
              'source_sha256': hashlib.sha256((root / 'train_matched_logical.py').read_bytes()).hexdigest(),
              'optimizer_step_performed': False, 'models': {}}

    for topology in ('EXPAND', 'REFINE'):
        exp.set_seed(2026)
        base, tokenizer = exp.load_base()
        model = exp.MatchedTopologyModel(base, topology).to(exp.device)
        # Keep dropout off while checking execution and gradients.
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        hashes = {name: digest_tensor(t) for name, t in {
            'token_embedding': base.transformer.wte.weight,
            'position_embedding': base.transformer.wpe.weight,
            'core_first_attention': model.core[0].attn.c_attn.weight,
            'coda_first_attention': model.coda[0].attn.c_attn.weight,
        }.items()}
        row = {'representative_initialization_hashes': hashes,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'adapter_parameters': sum(p.numel() for p in model.adapter.parameters()) if hasattr(model, 'adapter') else 0,
               'extra_norm_parameters': sum(p.numel() for p in model.core_norm.parameters()) if hasattr(model, 'core_norm') else 0,
               'attention_class': type(model.core[0].attn).__name__,
               'queries': []}
        q, prefix, target = exp.prepare_example(tokenizer, items[0])
        assert prefix.shape[1] == target.shape[1] == 1, 'Audit expects single-position output.'
        events = []
        handles = []
        def capture(name):
            def hook(module, inputs, output):
                events.append({'module': name, 'input_length': inputs[0].shape[1],
                               'output_length': output[0].shape[1]})
            return hook
        handles.append(model.core[-1].register_forward_hook(capture('last_core_block')))
        handles.append(model.coda[-1].register_forward_hook(capture('last_coda_block')))
        exp.set_seed(7001)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            logits = model(q, prefix)
            assert logits.shape[:2] == target.shape
            assert torch.isfinite(logits).all()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        row['events'] = events
        row['loss'] = loss.item()
        row['audit_forward_backward_seconds'] = time.perf_counter() - started
        row['peak_allocated_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        row['parameters_receiving_gradients'] = sum(p.numel() for p in model.parameters() if p.grad is not None)
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        row['core_gradient_norms'] = [
            float(block.attn.c_attn.weight.grad.float().norm())
            if block.attn.c_attn.weight.grad is not None else None
            for block in model.core]
        row['registered_but_unexecuted_gpt2_layers'] = list(range(8, len(base.transformer.h)))
        row['unused_layers_have_no_gradient'] = all(
            p.grad is None for block in base.transformer.h[8:] for p in block.parameters())
        del logits, loss
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            exp.set_seed(8001)
            a = model(q, prefix).detach()
            exp.set_seed(8001)
            b = model(q, prefix).detach()
            exp.set_seed(8002)
            c = model(q, prefix).detach()
            row['same_seed_max_logit_difference'] = float((a-b).abs().max())
            row['different_seed_max_logit_difference'] = float((a-c).abs().max())
            del a, b, c
            for item in items:
                qi, pi, ti = exp.prepare_example(tokenizer, item)
                exp.set_seed(9001)  # Shared randomness for the paired query diagnostic.
                scores = model(qi, pi)
                pred = scores.argmax(dim=-1)
                row['queries'].append({
                    'query_index': item['query_index'], 'target': item['answer'],
                    'prefix_tokens': pi[0].tolist(), 'target_tokens': ti[0].tolist(),
                    'prediction': tokenizer.decode(pred[0].tolist()),
                    'correct': bool(torch.equal(pred, ti)), 'input_tokens': qi.shape[1]})
        report['models'][topology] = row
        print(topology, json.dumps(row, indent=2), flush=True)
        del model, base, tokenizer, q, prefix, target, scores, pred, qi, pi, ti, handles
        gc.collect()
        torch.cuda.empty_cache()

    m = report['models']
    report['representative_backbone_initialization_matches'] = (
        m['EXPAND']['representative_initialization_hashes'] == m['REFINE']['representative_initialization_hashes'])
    report['scope'] = ('Checks execution, finite gradients and selected initialization tensors. '
                       'Predictions are before adaptation, not a learning or feasibility score. '
                       'No FLOP, generalization or architecture-only claim.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print('Saved:', out / 'summary.json', flush=True)
    print('AUDIT COMPLETE; no weights updated and no checkpoint saved.', flush=True)


if __name__ == '__main__':
    main()
