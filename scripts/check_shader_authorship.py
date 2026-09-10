"""Reject handwritten shader bodies outside the explicitly deferred FSR dependency.

Deferred dependencies are frozen and are not OrdinaryShade compliance.
Generated artifacts must identify their authoring source/generator in the inventory.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'ordinarylight/shaders/authorship.json'
SUFFIXES = {'.glsl', '.comp', '.wgsl', '.vert', '.frag', '.rgen', '.rmiss', '.rchit', '.rahit'}
# Function definitions, including structure-valued and scalar helpers. ABI-only
# declarations and compiler directives do not contain executable bodies.
SHADER_BODY = re.compile(r'\b(?:void|bool|u?int|float|[biu]?vec[234]|mat[234]|[A-Z]\w*)\s+\w+\s*\([^;{}]*\)\s*\{|\bfn\s+\w+\s*\([^;{}]*\)[^{]*\{')
DEFERRED = {'ordinarylight/shaders/fsr1_easu.glsl'}



def shader_sources(root=ROOT):
    paths = {p.relative_to(root).as_posix() for p in (root / 'ordinarylight').rglob('*')
             if p.is_file() and p.suffix in SUFFIXES}
    # The FSR build helper contains a native C++ vendor bridge, not shader logic.
    python_paths = [p for directory in ('ordinarylight', 'scripts')
                    for p in (root / directory).rglob('*.py')
                    if p.name not in {'check_shader_authorship.py', 'build_fsr2.py'}]
    for p in python_paths:
        tree = ast.parse(p.read_text())
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and SHADER_BODY.search(n.value) for n in ast.walk(tree)):
            paths.add(p.relative_to(root).as_posix())
    return paths


def legacy_digest(path):
    payload = path.read_bytes()
    if path.suffix == ".py":
        tree = ast.parse(payload.decode())
        payload = json.dumps([n.value for n in ast.walk(tree)
                              if isinstance(n, ast.Constant) and isinstance(n.value, str)
                              and SHADER_BODY.search(n.value)], ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def check(root=ROOT):
    inventory = json.loads((root / 'ordinarylight/shaders/authorship.json').read_text())
    entries = inventory['sources']
    errors = []
    actual = shader_sources(root)
    for path in sorted(actual - entries.keys()):
        errors.append(f'Unclassified shader source: {path}; author it in OrdinaryShade and register its generator')
    for path in sorted(entries.keys() - actual):
        errors.append(f'Stale inventory entry: {path}')
    for path in sorted(actual & entries.keys()):
        entry = entries[path]
        if path.endswith('.py'):
            errors.append(f'Executable shader body embedded in Python: {path}')
        if entry['status'] == 'deferred-third-party':
            if path not in DEFERRED:
                errors.append(f'Unapproved handwritten dependency: {path}')
            digest = legacy_digest(root / path)
            if digest != entry['sha256']:
                errors.append(f'Handwritten shader changed: {path}; migrate its logic to OrdinaryShade')
        elif entry['status'] == 'generated':
            for source in entry['sources']:
                if not (root / source).is_file():
                    errors.append(f'Missing OrdinaryShade generator/source: {source}')
        else:
            errors.append(f'Invalid authorship status: {path}')
    return errors


def main():
    errors = check()
    if errors:
        raise SystemExit('\n'.join(errors))
    entries = json.loads(INVENTORY.read_text())['sources']
    legacy = sum(e['status'] == 'deferred-third-party' for e in entries.values())
    print(f'Authorship inventory checked: {len(entries) - legacy} generated; {legacy} explicitly deferred third-party sources')


if __name__ == '__main__':
    main()
