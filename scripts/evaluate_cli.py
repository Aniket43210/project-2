"""Evaluation CLI skeleton for registered models.

Generates a JSON metrics stub and a Markdown model card using stored registry metadata.
Extend later with real dataset loading & metric computation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from stellar_platform.models.registry import ModelRegistry
from stellar_platform.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_dict,
    calibration_metrics,
    per_class_calibration,
    per_class_ece,
)
from stellar_platform.evaluation.calibration import expected_calibration_error
from stellar_platform.data.synthetic import generate_synthetic_spectra

try:  # optional TF
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore


def _load_artifact(meta: dict):
    art = meta.get('artifacts', {}).get('model')
    if not art:
        return None
    p = Path(art)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        return None
    if p.suffix.lower() == '.json':
        # dummy model descriptor
        try:
            payload = json.loads(p.read_text())
            classes = int(payload.get('classes', 3))
        except Exception:
            classes = 3
        class Dummy:
            def predict(self, x):  # type: ignore
                probs = np.random.rand(x.shape[0], classes)
                probs /= probs.sum(axis=1, keepdims=True)
                return probs
        return Dummy()
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(str(p))  # type: ignore
    except Exception:
        return None


def generate_real_metrics(family: str, meta: dict, n_eval: int = 128):
    # Determine number of classes
    params = meta.get('params', {})
    n_classes = int(params.get('num_classes', 3))
    length = int(params.get('length', 256))
    # Build evaluation dataset (synthetic for now)
    X, y = generate_synthetic_spectra(n=n_eval, length=length, n_classes=n_classes, seed=12345)
    model = _load_artifact(meta)
    if model is None:
        # Fallback random probs
        probs = np.random.rand(X.shape[0], n_classes)
        probs /= probs.sum(axis=1, keepdims=True)
    else:
        probs = model.predict(X)  # type: ignore
        # Ensure correct dimensionality
        if probs.shape[1] != n_classes:
            probs = probs[:, :n_classes]
    # Metrics
    cls_report = classification_metrics(y, probs)
    y_pred = np.argmax(probs, axis=1)
    acc = float(np.mean(y_pred == y))
    cm = confusion_matrix_dict(y, probs)
    # Focus class calibration (class 0) plus per-class curves
    calib = calibration_metrics(y, probs, focus_class=0, n_bins=10, method="quantile")
    calib_all = per_class_calibration(y, probs, n_bins=10, method="quantile")
    ece_per_class = per_class_ece(y, probs, n_bins=10, method="quantile")
    ece = expected_calibration_error(probs, y)
    return {
        'accuracy': acc,
        'macro_f1': cls_report.macro_f1,
        'macro_precision': cls_report.macro_precision,
        'macro_recall': cls_report.macro_recall,
        'mcc': cls_report.mcc,
        'roc_auc_macro': cls_report.roc_auc_macro,
        'pr_auc_macro': cls_report.pr_auc_macro,
        'per_class': cls_report.per_class,
        'confusion_matrix': cm,
        'calibration': calib,
        'calibration_per_class': calib_all,
        'ece': ece,
        'ece_per_class': ece_per_class,
        'n_samples': int(X.shape[0]),
        'generated_at': datetime.utcnow().isoformat() + 'Z'
    }


def _sanitize_for_json(obj):
    """Recursively replace NaN/inf with None so JSON is standards-compliant.

    Python's json.dumps by default allows NaN/Infinity which is non-RFC8259.
    We explicitly clean them so downstream strict parsers won't fail.
    """
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _sanitize_for_json(v) for v in obj ]
    return obj


def write_model_card(output_dir: Path, family: str, version: str, metadata: dict, metrics: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    card = output_dir / f"{family}_{version}_model_card.md"
    lines = [
        f"# Model Card: {family} ({version})",
        "",
        "## Overview",
        f"Family: {family}",
        f"Version: {version}",
        f"Registered At: {metadata.get('registered_at')}",
        f"Artifacts: {json.dumps(metadata.get('artifacts', {}), indent=2)}",
        "",
        "## Params",
        '```json',
        json.dumps(metadata.get('params', {}), indent=2),
        '```',
        "",
        "## Metrics",
        '```json',
        json.dumps(metrics, indent=2),
        '```',
        "",
        "## Calibration",
        '```json',
        json.dumps(metadata.get('calibrator', {}), indent=2),
        '```',
        "",
        "## Notes",
        "- This is an auto-generated placeholder model card. Replace metrics with real evaluation results.",
    ]
    card.write_text("\n".join(lines))
    return card


def main():  # pragma: no cover - integration
    parser = argparse.ArgumentParser(description="Evaluate a registered model and produce reports")
    parser.add_argument("family", help="Model family name (e.g., spectral_cnn)")
    parser.add_argument("--version", help="Specific version (defaults to latest)")
    parser.add_argument("--output-dir", default="evaluations", help="Directory for outputs")
    args = parser.parse_args()

    registry = ModelRegistry()
    version = args.version or registry.get_latest_version(args.family)
    if version is None:
        raise SystemExit(f"No versions found for family {args.family}")
    meta = registry.get_metadata(args.family, version)
    if meta is None:
        raise SystemExit(f"Metadata not found for {args.family}:{version}")

    metrics = generate_real_metrics(args.family, meta)
    metrics = _sanitize_for_json(metrics)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.family}_{version}_metrics.json"
    # use allow_nan=False to enforce we didn't miss anything
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False))

    card_path = write_model_card(out_dir, args.family, version, meta, metrics)
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote model card to {card_path}")


if __name__ == "__main__":
    main()
