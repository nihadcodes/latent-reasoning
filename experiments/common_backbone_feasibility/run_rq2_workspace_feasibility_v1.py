"""RQ2 common-update feasibility prototype, v1. Requires local GPT-2.

Custom baselines, not original Coconut/Huginn/Ouro reproductions. All use the
same pretrained GPT-2 initialization, full stack, head, data and optimizer rule.
Each pass recomputes raw question context (no cache). Workspace starts with the
EOS token embedding. EXPAND preserves it and appends the latest update; REFINE
replaces the fixed workspace. GPT-2 adds explicit position embeddings each pass.
A final shared GPT-2 readout sees the question, workspace and ### prefix.

Same update counts are NOT matched FLOPs. This first-stage test measures
execution and learning, not hybrid superiority or structural generalization.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def make_data(groups, seed, prefix, forbidden=None):
    rng = random.Random(seed)
    used = set() if forbidden is None else set(forbidden)
    rows, signatures = [], set()
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for group in range(groups):
        while True:
            a, b, c, p, q, r = rng.sample(alphabet, 6)
            edges = [(a, b), (b, c), (p, q), (q, r)]
            signature = tuple(sorted(edges))
            if signature not in used:
                used.add(signature)
                signatures.add(signature)
                break
        rng.shuffle(edges)
        facts = ' '.join(f'{u} -> {v}.' for u, v in edges)
        for query_index, (start, answer) in enumerate(((a, c), (p, r))):
            row = {'base_id': f'{prefix}_{group:04d}', 'query_index': query_index,
                   'condition': 'original', 'edges': edges, 'start': start,
                   'hops': 2, 'answer': answer,
                   'question': f'Directed edges: {facts}\nStart at {start}. Follow exactly 2 directed edges. Which node do you reach?'}
            successor = dict(edges)
            node = start
            for _ in range(row['hops']):
                node = successor[node]
            assert node == answer
            rows.append(row)
    return rows, signatures


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    import torch
    import torch.nn.functional as F
    from torch import nn
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'tiny'], default='smoke')
    parser.add_argument('--model-dir', type=Path, default=Path('../gpt2_model'))
    parser.add_argument('--output-root', type=Path, default=Path('rq2_workspace_feasibility'))
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--train-groups', type=int, default=16)
    parser.add_argument('--dev-groups', type=int, default=64)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Gradient accumulation; each example is forwarded separately.')
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save-weights', action='store_true')
    args = parser.parse_args()
    assert args.steps >= 1
    assert min(args.train_groups, args.dev_groups, args.updates, args.batch_size, args.eval_every) >= 1
    assert math.isfinite(args.lr) and args.lr > 0
    assert torch.cuda.is_available(), 'Run on an allocated CUDA node.'
    device = torch.device('cuda')
    # FP32 throughout; disable TF32 for a clear initial numerical convention.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def seed(value):
        random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
    seed(args.seed)
    out = args.output_root / (args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    train, train_sigs = make_data(args.train_groups, args.seed + 1000, 'train')
    dev, dev_sigs = make_data(args.dev_groups, args.seed + 2000, 'dev', train_sigs)
    assert not train_sigs & dev_sigs
    assert not {x['question'] for x in train} & {x['question'] for x in dev}
    dump(out / 'train.json', train)
    dump(out / 'development.json', dev)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir.resolve()), local_files_only=True)
    prefix_ids = tokenizer.encode('###', add_special_tokens=False)
    assert len(prefix_ids) == 1, prefix_ids
    assert tokenizer.eos_token_id is not None
    template = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir.resolve()), local_files_only=True,
        attn_implementation='eager', torch_dtype=torch.float32)
    # Identical, dropout-free conditions for this feasibility stage.
    for module in template.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    template.config.use_cache = False
    template.eval()
    assert template.config.model_type == 'gpt2'
    assert all(type(block.attn).__name__ == 'GPT2Attention' for block in template.transformer.h)
    def state_hash(model):
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update(name.encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial_hash = state_hash(template)
    counts = sum(p.numel() for p in template.parameters())

    class WorkspaceModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.last_trace = None

        def pass_states(self, embeddings):
            length = embeddings.shape[1]
            assert length <= self.base.config.n_positions
            positions = torch.arange(length, device=device).unsqueeze(0)
            return self.base.transformer(
                inputs_embeds=embeddings, position_ids=positions,
                attention_mask=torch.ones((1, length), device=device, dtype=torch.long),
                use_cache=False, output_hidden_states=False, return_dict=True,
            ).last_hidden_state

        def forward(self, question_ids, strategy, steps, capture=False):
            assert strategy in ('DIRECT', 'EXPAND', 'REFINE')
            if strategy == 'DIRECT':
                assert steps == 0
            context = self.base.transformer.wte(question_ids)
            qlen = context.shape[1]
            eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
            workspace = self.base.transformer.wte(eos)
            trace = []
            # Raw context is reset each pass. Only the workspace is recurrent.
            for t in range(steps):
                old = workspace
                states = self.pass_states(torch.cat([context, old], dim=1))
                updates = states[:, qlen:, :]
                if strategy == 'EXPAND':
                    workspace = torch.cat([old, updates[:, -1:, :]], dim=1)
                else:
                    workspace = updates
                if capture:
                    preservation_error = float((workspace[:, :old.shape[1], :] - old).abs().max()) if strategy == 'EXPAND' else None
                    trace.append({'event': t + 1, 'input_length': qlen + old.shape[1],
                                  'workspace_before': old.shape[1], 'workspace_after': workspace.shape[1],
                                  'preserved_prefix_max_error': preservation_error})
            prefix = self.base.transformer.wte(torch.tensor([prefix_ids], device=device))
            readout_inputs = torch.cat([context, workspace, prefix], dim=1)
            final = self.pass_states(readout_inputs)[:, -1, :]
            logits = self.base.lm_head(final)
            if capture:
                self.last_trace = {'events': trace, 'question_length': qlen,
                                   'readout_length': readout_inputs.shape[1],
                                   'final_workspace_length': workspace.shape[1]}
            return logits

    def encode(row):
        qids = tokenizer.encode(row['question'] + '\n', add_special_tokens=False)
        tids = tokenizer.encode(row['answer'], add_special_tokens=False)
        assert len(tids) == 1, (row['answer'], tids)
        return (torch.tensor([qids], device=device), torch.tensor(tids, device=device))
    train_tensors = [encode(row) for row in train]
    dev_tensors = [encode(row) for row in dev]

    @torch.no_grad()
    def evaluate(model, rows, tensors, strategy, steps):
        model.eval()
        results, losses = [], []
        for row, (question, target) in zip(rows, tensors):
            logits = model(question, strategy, steps)
            losses.append(float(F.cross_entropy(logits, target)))
            predicted_id = int(logits.argmax(dim=-1).item())
            results.append({'base_id': row['base_id'], 'query_index': row['query_index'],
                            'target': row['answer'], 'prediction': tokenizer.decode([predicted_id]),
                            'prediction_token_id': predicted_id, 'correct': predicted_id == target.item()})
        pairs = defaultdict(list)
        for row in results:
            pairs[row['base_id']].append(row)
        assert all(len(pair) == 2 and {r['query_index'] for r in pair} == {0, 1}
                   and pair[0]['target'] != pair[1]['target'] for pair in pairs.values())
        summary = {'loss': sum(losses) / len(losses), 'correct': sum(r['correct'] for r in results),
                   'records': len(results), 'accuracy': sum(r['correct'] for r in results) / len(results),
                   'pairs': len(pairs),
                   'both_queries_correct': sum(all(r['correct'] for r in pair) for pair in pairs.values()),
                   'same_prediction_pairs': sum(pair[0]['prediction_token_id'] == pair[1]['prediction_token_id'] for pair in pairs.values()),
                   'prediction_counts': dict(Counter(r['prediction'] for r in results))}
        return {'summary': summary, 'results': results}

    manifest = {'experiment': 'RQ2 common-update execution and learning feasibility v1',
                'mode': args.mode, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
                'precision': 'FP32 parameters, gradients and AdamW moments; TF32 disabled',
                'initial_state_sha256': initial_hash, 'parameter_count': counts,
                'backbone_layers': len(template.transformer.h), 'attention': 'eager causal GPT-2',
                'dropout': 0.0, 'cache': False,
                'answer_metric': 'Exact answer-token accuracy over full vocabulary; answer not in input suffix',
                'workspace_initialization': 'One EOS embedding, identical deterministic rule for all strategies',
                'positions': 'Contiguous position IDs for every full pass; GPT-2 adds absolute position embeddings',
                'data_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ('train.json', 'development.json')},
                'limitations': ['Custom GPT-2 workspace baselines, not original model reproductions.',
                                'Update counts are not matched FLOPs; no RQ2 performance conclusion.',
                                'Two-chain task tests learning of a narrow family; graph topologies are isomorphic.',
                                'Development graphs exclude identical labelled edge sets, not isomorphic structures.',
                                'Final readout also sees input context; learned use of latents requires later ablation.',
                                'One training seed; inference initialization is deterministic.']}
    dump(out / 'configuration.json', manifest)
    print('OUTPUT DIRECTORY:', out.resolve(), flush=True)
    print('PARAMETERS:', counts, 'LAYERS:', len(template.transformer.h), flush=True)
    all_results = {}
    # Fix training order in advance and reuse it for every strategy.
    order_rng = random.Random(args.seed + 3000)
    training_batches = []
    order, cursor = [], 0
    for _ in range(args.updates):
        batch = []
        for _ in range(args.batch_size):
            if cursor == len(order):
                order = list(range(len(train)))
                order_rng.shuffle(order)
                cursor = 0
            batch.append(order[cursor])
            cursor += 1
        training_batches.append(batch)
    dump(out / 'training_batches.json', training_batches)

    for strategy, steps in (('DIRECT', 0), ('EXPAND', args.steps), ('REFINE', args.steps)):
        seed(args.seed)
        base = copy.deepcopy(template)
        assert state_hash(base) == initial_hash
        model = WorkspaceModel(base).to(device)
        folder = out / strategy
        folder.mkdir()
        model.eval()
        q, target = train_tensors[0]
        # Check causal attention using a future-input perturbation on the actual stack.
        with torch.no_grad():
            inp = model.base.transformer.wte(q)
            changed = inp.clone()
            changed[:, -1, :] += 0.5 * torch.sin(torch.arange(inp.shape[-1], device=device).float())
            early_a = model.pass_states(inp)[:, :-1, :]
            early_b = model.pass_states(changed)[:, :-1, :]
            causal_error = float((early_a - early_b).abs().max())
            assert causal_error <= 1e-6, ('Causal attention check failed', causal_error)
            a = model(q, strategy, steps, capture=True)
            trace = model.last_trace
            b = model(q, strategy, steps)
            repeat_error = float((a - b).abs().max())
            assert repeat_error <= 1e-6
        del a, b, inp, changed, early_a, early_b
        expected_workspace = 1 + steps if strategy == 'EXPAND' else 1
        assert trace['final_workspace_length'] == expected_workspace
        for event in trace['events']:
            if strategy == 'EXPAND':
                assert event['workspace_after'] == event['workspace_before'] + 1
                assert event['preserved_prefix_max_error'] == 0.0
            else:
                assert event['workspace_after'] == event['workspace_before']
        # Backward check with labels passed only to the loss.
        logits = model(q, strategy, steps)
        assert logits.shape == (1, model.base.config.vocab_size)
        assert torch.isfinite(logits).all()
        loss = F.cross_entropy(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        layer_norms = [float(block.attn.c_attn.weight.grad.float().norm()) for block in model.base.transformer.h]
        assert all(math.isfinite(v) and v > 0 for v in layer_norms), layer_norms
        audit = {'passed': True, 'strategy': strategy, 'steps': steps,
                 'initial_state_sha256': initial_hash,
                 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'trace': trace, 'causal_future_perturbation_error': causal_error,
                 'same_input_repeat_error': repeat_error, 'loss': float(loss),
                 'per_layer_attention_gradient_norms': layer_norms,
                 'question_token_count': q.shape[1], 'answer_prefix_ids': prefix_ids,
                 'target_token_id': int(target.item()), 'target_is_loss_only': True,
                 'optimizer_step_performed_in_audit': False}
        dump(folder / 'execution_audit.json', audit)
        print('EXECUTION_AUDIT', strategy, json.dumps(audit), flush=True)
        model.zero_grad(set_to_none=True)
        del logits, loss
        initial = evaluate(model, train, train_tensors, strategy, steps)
        dump(folder / 'initial_train.json', initial)
        print('INITIAL', strategy, json.dumps(initial['summary']), flush=True)
        if args.mode == 'tiny':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            history = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for update, batch in enumerate(training_batches, 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lr = args.lr * min(update / 10, 1.0)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                batch_loss = 0.0
                for index in batch:
                    qi, ti = train_tensors[index]
                    scores = model(qi, strategy, steps)
                    loss = F.cross_entropy(scores, ti)
                    assert torch.isfinite(loss), (strategy, update)
                    (loss / len(batch)).backward()
                    batch_loss += float(loss.detach()) / len(batch)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if update == 1:
                    assert all(state['exp_avg'].dtype == torch.float32 and state['exp_avg_sq'].dtype == torch.float32
                               for state in optimizer.state.values() if 'exp_avg' in state)
                if update % args.eval_every == 0 or update == args.updates:
                    evaluation = evaluate(model, train, train_tensors, strategy, steps)
                    torch.cuda.synchronize()
                    row = {'update': update, 'last_batch_loss': batch_loss,
                           'preclip_gradient_norm': float(grad_norm), 'learning_rate': lr,
                           'train': evaluation['summary'],
                           'elapsed_seconds_including_evaluation': time.perf_counter() - start,
                           'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3}
                    history.append(row)
                    dump(folder / 'history.json', history)
                    print('TINY', strategy, json.dumps(row), flush=True)
            # Same fixed number of updates for all strategies; no early stopping.
            final_train = evaluate(model, train, train_tensors, strategy, steps)
            development = evaluate(model, dev, dev_tensors, strategy, steps)
            dump(folder / 'final_train.json', final_train)
            dump(folder / 'development.json', development)
            final = {'execution_passed': True, 'updates': args.updates,
                     'train': final_train['summary'], 'development': development['summary'],
                     'tiny_set_fully_fit': final_train['summary']['correct'] == len(train),
                     'all_training_query_pairs_correct': final_train['summary']['both_queries_correct'] == args.train_groups,
                     'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3}
            if args.save_weights:
                torch.save({'model_state_dict': model.state_dict(), 'strategy': strategy,
                            'steps': steps, 'configuration': manifest}, folder / 'final_weights.pt')
            del optimizer, scores, loss, group
        else:
            final = {'execution_passed': True, 'initial_train': initial['summary'],
                     'training_performed': False}
        all_results[strategy] = final
        dump(out / 'summary.json', all_results)
        print('FINAL', strategy, json.dumps(final), flush=True)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    print('COMPLETE:', out.resolve(), flush=True)
    print('This is execution/learning feasibility, not a matched-compute comparison.', flush=True)


if __name__ == '__main__':
    main()
