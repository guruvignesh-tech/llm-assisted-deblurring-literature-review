"""Bind a stage's checkpoints to its exact input and script configuration."""
import hashlib
import json
from pathlib import Path


def bind_checkpoint(stage, input_path, script_path, checkpoint_paths):
    manifest = Path(f'{stage}_input_manifest.json')
    expected = {
        'schema_version': 1,
        'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(script_path).read_bytes()).hexdigest(),
    }
    if manifest.exists():
        actual = json.loads(manifest.read_text(encoding='utf-8'))
        if actual != expected:
            raise ValueError(f'{stage}: input or script changed. Use a fresh run directory; do not reuse these checkpoints.')
    else:
        existing = [str(p) for p in checkpoint_paths if Path(p).exists()]
        if existing:
            raise ValueError(f'{stage}: unverified checkpoints found: {existing}. Preserve them and start in a fresh directory.')
        tmp = manifest.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(expected, indent=2)+'\n', encoding='utf-8')
        tmp.replace(manifest)
    return expected
