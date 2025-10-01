"""Migrate legacy .h5 model artifacts in registry to native .keras format and update metadata.

This script walks the filesystem registry and for each model version whose
artifacts include a 'model' ending with .h5, it will:
  - load the model via tf.keras
  - save it to a new .keras file (family_version.keras)
  - update metadata 'artifacts.model' and 'artifact_hashes.model'
  - optionally remove the old .h5 file

Run:
  python scripts/migrate_keras_format.py --registry registry --artifacts-dir artifacts --remove-old
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
from typing import Dict

import tensorflow as tf  # type: ignore


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def migrate_group(h5_path: Path, refs: Dict[Path, list[tuple[Path, str]]], artifacts_dir: Path, remove_old: bool = False) -> int:
    # Load once
    model = None
    errors = []
    for kwargs in ({}, {"compile": False}):
        try:
            model = tf.keras.models.load_model(str(h5_path), **kwargs)
            break
        except Exception as e:
            errors.append(str(e))
            model = None
    if model is None:
        print(f"ERROR: Failed to load {h5_path}: {' | '.join(errors)}")
        return 0

    migrated = 0
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for family_dir, version in refs[h5_path]:
        vdir = family_dir / version
        meta_path = vdir / 'metadata.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        new_name = f"{family_dir.name}_{version}.keras"
        new_path = artifacts_dir / new_name
        try:
            model.save(str(new_path))
        except Exception as e:
            print(f"ERROR: Failed to save .keras to {new_path}: {e}")
            continue
        rel_new = str(new_path.relative_to(Path.cwd())) if new_path.is_absolute() else str(new_path)
        meta.setdefault('artifacts', {})['model'] = rel_new
        meta.setdefault('artifact_hashes', {})['model'] = sha256_file(new_path)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Migrated {family_dir.name}:{version} -> {new_path}")
        migrated += 1

    if remove_old:
        try:
            h5_path.unlink()
            print(f"Removed old {h5_path}")
        except Exception as e:
            print(f"WARN: Could not remove old {h5_path}: {e}")
    return migrated


def main():
    ap = argparse.ArgumentParser(description="Migrate registry .h5 artifacts to .keras")
    ap.add_argument('--registry', default='registry', help='Registry root directory')
    ap.add_argument('--artifacts-dir', default='artifacts', help='Directory to store converted .keras files')
    ap.add_argument('--remove-old', action='store_true', help='Remove old .h5 files after successful migration')
    args = ap.parse_args()

    reg = Path(args.registry)
    arts = Path(args.artifacts_dir)
    if not reg.exists():
        raise SystemExit(f"Registry path not found: {reg}")

    # Build reference map: h5 absolute path -> list of (family_dir, version)
    refs: Dict[Path, list[tuple[Path, str]]] = {}
    for family_dir in [p for p in reg.iterdir() if p.is_dir()]:
        for version_dir in [p for p in family_dir.iterdir() if p.is_dir() and p.name != 'latest']:
            meta_path = version_dir / 'metadata.json'
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            artifacts = meta.get('artifacts', {})
            model_path_str = artifacts.get('model')
            if not model_path_str or not model_path_str.lower().endswith('.h5'):
                continue
            p = Path(model_path_str)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            refs.setdefault(p, []).append((family_dir, version_dir.name))

    migrated = 0
    # Pre-scan available .keras per family for fallback
    keras_by_family: Dict[str, list[Path]] = {}
    for p in arts.glob('*.keras'):
        fam = p.name.rsplit('_', 1)[0]
        keras_by_family.setdefault(fam, []).append(p)

    for h5_path, lst in refs.items():
        if h5_path.exists():
            migrated += migrate_group(h5_path, refs, arts, remove_old=args.remove_old)
            continue
        # Fallback: update metadata to use any existing .keras of same family
        fam = lst[0][0].name if lst else 'unknown'
        candidates = keras_by_family.get(fam, [])
        if not candidates:
            print(f"WARN: Missing file {h5_path}; no .keras candidates for family {fam}; leaving as-is for {len(lst)} refs")
            continue
        target = candidates[0]
        for family_dir, version in lst:
            vdir = family_dir / version
            meta_path = vdir / 'metadata.json'
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            rel_new = str(target.relative_to(Path.cwd())) if target.is_absolute() else str(target)
            meta.setdefault('artifacts', {})['model'] = rel_new
            meta.setdefault('artifact_hashes', {})['model'] = sha256_file(target)
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"UPDATED (fallback) {family_dir.name}:{version} -> {target}")

    print(f"Migration complete. Versions migrated: {migrated}")


if __name__ == '__main__':  # pragma: no cover
    main()
