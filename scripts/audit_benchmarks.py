"""CPU-only solver and saved-artifact checks; does not execute a language model."""
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

def check_rows(rows, count, conditions):
    assert len(rows) == count, (len(rows), count)
    groups = defaultdict(list)
    for row in rows:
        edges = row['edges']
        nxt = dict(edges)
        assert len(nxt) == len(edges), 'Ambiguous outgoing edges'
        node = row['start']
        for _ in range(row['hops']):
            node = nxt[node]
        assert node == row['answer']
        assert sorted(re.findall(r'([A-Z]) -> ([A-Z])\.', row['question'])) == sorted(map(tuple, edges))
        query = re.search(r'Start at ([A-Z])\. Follow exactly (\d+) directed edges', row['question'])
        assert query and query.group(1) == row['start'] and int(query.group(2)) == row['hops']
        groups[(row['base_id'], row['condition'])].append(row)
    by_base = defaultdict(set)
    for (base, condition), pair in groups.items():
        assert len(pair) == 2 and {r['query_index'] for r in pair} == {0, 1}
        assert pair[0]['start'] != pair[1]['start']
        assert pair[0]['answer'] != pair[1]['answer']
        assert sorted(map(tuple, pair[0]['edges'])) == sorted(map(tuple, pair[1]['edges']))
        by_base[base].add(condition)
    assert all(c == conditions for c in by_base.values())
    return {'records': count, 'base_groups': len(by_base), 'targets_solver_verified': count}

def load_module(path):
    spec = importlib.util.spec_from_file_location('reviewed_pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    root = Path(__file__).resolve().parents[1]
    report = {}
    prompts = {}
    cond = {'original', 'relabeling', 'path_change', 'answer_change'}
    for split, count in [('train', 1600), ('validation', 320), ('test', 800)]:
        rows = json.loads((root / 'data/controlled_v3_twochains' / (split + '.json')).read_text())
        report[split] = check_rows(rows, count, cond)
        prompts[split] = {r['question'] for r in rows}
    assert not (prompts['train'] & prompts['validation'])
    assert not (prompts['train'] & prompts['test'])
    assert not (prompts['validation'] & prompts['test'])
    report['split_control'] = 'No exact question overlap; isomorphic structures may overlap.'
    rows = json.loads((root / 'data/relevance_control_20261011_171002/diagnostic.json').read_text())
    report['relevance'] = check_rows(rows, 400, {
        'original', 'path_chain_0', 'path_chain_1', 'endpoint_chain_0', 'endpoint_chain_1'})
    for p in (root / 'data/relevance_control_20261011_171002').glob('*audit.json'):
        saved = json.loads(p.read_text())['summary']
        assert saved['all_matched'] is True
        report[p.name] = 'Saved audit reports matched edits; tokenizer was not rerun.'
    run = root / 'results/rq2_feasibility/tiny_20261024_022646_327722'
    cfg = json.loads((run / 'configuration.json').read_text())
    script = root / 'experiments/common_backbone_feasibility/run_rq2_workspace_feasibility_v1.py'
    module = load_module(script)
    a = cfg['arguments']
    train, used = module.make_data(a['train_groups'], a['seed'] + 1000, 'train')
    dev, _ = module.make_data(a['dev_groups'], a['seed'] + 2000, 'dev', used)
    for name, data in [('train.json', train), ('development.json', dev)]:
        value = hashlib.sha256(json.dumps(data, indent=2).encode()).hexdigest()
        assert value == cfg['data_sha256'][name], (name, 'Generated data differs from recorded run')
    report['rq2_data'] = 'Regenerated train/development bytes match both recorded SHA256 values.'
    summary = json.loads((run / 'summary.json').read_text())
    for strategy, item in summary.items():
        d = item['development']
        assert d['correct'] / d['records'] == d['accuracy']
        assert d['records'] == 256 and d['pairs'] == 128
        report[strategy] = {'recorded_accuracy': d['accuracy'], 'correct': d['correct']}
    print(json.dumps(report, indent=2))
    print('BENCHMARK AUDIT PASSED: solver, split, saved audit and RQ2 data regeneration checks only.')

if __name__ == '__main__':
    main()
