# Launch the unchanged RQ2 script with explicit local paths.
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('local/rq2_runs'))
    args, extra = p.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    forbidden = ('--model-dir', '--output-root')
    if any(x.split('=')[0] in forbidden for x in extra):
        p.error('Set model/output paths through the launcher, not the extra arguments.')
    model = args.model_dir.resolve()
    if not (model / 'config.json').is_file():
        p.error('Model directory must contain config.json and local GPT-2 files.')
    root = Path(__file__).resolve().parents[1]
    target = root / 'experiments/common_backbone_feasibility/run_rq2_workspace_feasibility_v1.py'
    command = [sys.executable, str(target), '--model-dir', str(model),
               '--output-root', str(args.output_root.resolve()), *extra]
    return subprocess.call(command)

if __name__ == '__main__':
    raise SystemExit(main())
