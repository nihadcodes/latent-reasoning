"""Run on the lab server to collect the remaining review inputs, excluding weights."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nihad-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path.cwd())
    args = p.parse_args()
    root = args.nihad_root.resolve()
    project = root / 'coconut/mentor_experiment'
    paths = [
        project / 'controlled_v3_twochains/train.json',
        project / 'controlled_v3_twochains/validation.json',
        project / 'controlled_v3_twochains/test.json',
        project / 'relevance_control_20261011_171002/diagnostic.json',
        root / 'SIM-CoT_review/Coconut/coconut.py',
        root / 'SIM-CoT_review/Coconut/args/gsm_simcot_eval.yaml',
        root / 'SIM-CoT_review/LICENSE',
        root / 'reasoning_by_superposition_review/data/prosqa_test_graph_4_coconut.json',
    ]
    for folder in ['huginn_0125_torch24_compat', 'ouro_1_4b', 'simcot_gpt2_coconut']:
        model = root / folder
        if model.is_dir():
            paths.extend(model.glob('*.py'))
            for name in ['config.json', 'download_revision.json']:
                paths.append(model / name)
    allowed = {'summary.json', 'configuration.json', 'paired_conditions.csv',
               'paired_contrasts.csv', 'group_summary.csv', 'feedback_checks.json',
               'coconut_token_audit.json', 'huginn_token_audit.json',
               'ouro_token_audit.json', 'simcot_token_audit.json'}
    for folder in [project / 'mentor_state_comparison',
                   project / 'rq2_workspace_feasibility',
                   root / 'reasoning_by_superposition_review/review_runs',
                   project / 'relevance_control_20261011_171002']:
        if folder.is_dir():
            paths.extend(x for x in folder.rglob('*') if x.name in allowed)
    found, missing, oversized = [], [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / ('thesis_missing_inputs_' +
                             datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.zip')
    with ZipFile(out, 'x', ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            if not path.is_file():
                missing.append(str(path.relative_to(root)))
                continue
            if path.stat().st_size > 5 * 1024**2:
                oversized.append(str(path.relative_to(root)))
                continue
            rel = str(path.relative_to(root))
            data = path.read_bytes()
            z.writestr(rel, data)
            found.append({'path': rel, 'size_bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})
        z.writestr('collection_manifest.json', json.dumps(
            {'files': found, 'missing': missing, 'oversized': oversized}, indent=2))
    print('Created:', out)
    print('Files:', len(found), 'MiB:', round(out.stat().st_size / 1024**2, 2))
    print('Missing:', len(missing), 'Oversized skipped:', len(oversized))
    print('Review the collected text files before sharing. No model weights selected.')

if __name__ == '__main__':
    main()
