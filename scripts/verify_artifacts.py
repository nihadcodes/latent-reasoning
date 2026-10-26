# Check packaging, syntax and source-copy integrity without model imports.
import ast
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    files = [p for p in root.rglob('*') if p.is_file()
             and '.git' not in p.relative_to(root).parts
             and '__pycache__' not in p.relative_to(root).parts]
    for p in files:
        rel = str(p.relative_to(root))
        if p.stat().st_size > 10 * 1024**2:
            errors.append('Large file: ' + rel)
        if p.suffix in {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.zip'}:
            errors.append('Excluded artifact: ' + rel)
        if p.suffix == '.py':
            try:
                ast.parse(p.read_text())
            except (SyntaxError, UnicodeError) as e:
                errors.append('Syntax/encoding: ' + rel + ': ' + str(e))
    items = json.loads((root / 'docs/source_manifest.json').read_text())
    items += json.loads((root / 'docs/artifact_manifest.json').read_text())
    for item in items:
        p = root / item['repository_path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Source copy differs: ' + item['repository_path'])
    print(json.dumps({'files_checked': len(files), 'errors': errors,
                      'scope': 'packaging and source integrity only; no GPU reproduction'}, indent=2))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
