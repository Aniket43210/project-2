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
from stellar_platform.evaluation.ensembles import average_probs, logit_average
from stellar_platform.evaluation.conformal import compute_conformal_threshold, empirical_coverage
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


def _predict_from_meta(meta: dict, X: np.ndarray, n_classes: int) -> np.ndarray:
    model = _load_artifact(meta)
    if model is None:
        probs = np.random.rand(X.shape[0], n_classes)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
    probs = model.predict(X)  # type: ignore
    if probs.shape[1] != n_classes:
        probs = probs[:, :n_classes]
    return probs


def generate_real_metrics(
    family: str,
    meta: dict,
    n_eval: int = 128,
    ensemble_families: list[str] | None = None,
    ensemble_versions: list[str] | None = None,
    ensemble_method: str = "prob",
    conformal_alpha: float | None = None,
    conformal_cal_fraction: float = 0.3,
):
    # Determine number of classes
    params = meta.get('params', {})
    n_classes = int(params.get('num_classes', 3))
    length = int(params.get('length', 256))
    # Build evaluation dataset (synthetic for now)
    X, y = generate_synthetic_spectra(n=n_eval, length=length, n_classes=n_classes, seed=12345)
    # Single model or ensemble
    if not ensemble_families:
        probs = _predict_from_meta(meta, X, n_classes)
    else:
        # Load registry to fetch additional metas
        from stellar_platform.models.registry import ModelRegistry
        reg = ModelRegistry()
        metas = []
        for i, fam in enumerate(ensemble_families):
            ver = None
            if ensemble_versions and i < len(ensemble_versions):
                ver = ensemble_versions[i]
            ver = ver or reg.get_latest_version(fam)
            if ver is None:
                continue
            m = reg.get_metadata(fam, ver)
            if m is not None:
                metas.append(m)
        prob_list = [_predict_from_meta(m, X, n_classes) for m in metas]
        if not prob_list:
            probs = _predict_from_meta(meta, X, n_classes)
        else:
            if ensemble_method == "logit":
                probs = logit_average(prob_list)
            else:
                probs = average_probs(prob_list)
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
    result = {
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
    # Optional split conformal report using a calibration subset
    if conformal_alpha is not None and 0 < conformal_alpha < 1:
        n = X.shape[0]
        n_cal = max(1, int(n * conformal_cal_fraction))
        # regenerate independently to avoid leakage in synthetic eval: use same params
        X2, y2 = generate_synthetic_spectra(n=n, length=length, n_classes=n_classes, seed=12346)
        probs2 = _predict_from_meta(meta, X2, n_classes) if not ensemble_families else (
            logit_average([_predict_from_meta(m, X2, n_classes) for m in metas]) if ensemble_method == "logit" else average_probs([_predict_from_meta(m, X2, n_classes) for m in metas])
        )
        y_cal, y_test = y2[:n_cal], y2[n_cal:]
        p_cal, p_test = probs2[:n_cal], probs2[n_cal:]
        q = compute_conformal_threshold(y_cal, p_cal, alpha=conformal_alpha)
        cov = empirical_coverage(y_test, p_test, q)
        result['conformal'] = {
            'alpha': float(conformal_alpha),
            'q_hat': float(q),
            'empirical_coverage': float(cov),
            'n_cal': int(n_cal),
            'n_test_for_coverage': int(len(y_test)),
        }
    return result


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
        "## Reproducibility",
        f"Seed: {metadata.get('seed', 'N/A')}",
        '```json',
        json.dumps(metadata.get('calibrator', {}), indent=2),
        '```',
        "",
        "## Conformal (if enabled)",
        *([] if 'conformal' not in metrics else [
            f"Alpha: {metrics['conformal'].get('alpha')}",
            f"q_hat: {metrics['conformal'].get('q_hat')}",
            f"Empirical coverage: {metrics['conformal'].get('empirical_coverage')}",
            f"Calibration size: {metrics['conformal'].get('n_cal')}"
        ]),
        "",
        "## Metrics",
        '```json',
        json.dumps(metrics, indent=2),
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
    parser.add_argument("--families", help="Comma-separated additional families to ensemble with", default="")
    parser.add_argument("--versions", help="Comma-separated versions matching --families (optional)", default="")
    parser.add_argument("--ensemble-method", choices=["prob", "logit"], default="prob")
    parser.add_argument("--conformal-alpha", type=float, help="Enable split conformal with given alpha (e.g., 0.1)")
    parser.add_argument("--conformal-cal-fraction", type=float, default=0.3, help="Calibration fraction for conformal split")
    args = parser.parse_args()

    registry = ModelRegistry()
    version = args.version or registry.get_latest_version(args.family)
    if version is None:
        raise SystemExit(f"No versions found for family {args.family}")
    meta = registry.get_metadata(args.family, version)
    if meta is None:
        raise SystemExit(f"Metadata not found for {args.family}:{version}")

    fams = [f.strip() for f in args.families.split(',') if f.strip()]
    vers = [v.strip() for v in args.versions.split(',') if v.strip()]
    metrics = generate_real_metrics(
        args.family,
        meta,
        ensemble_families=fams or None,
        ensemble_versions=vers or None,
        ensemble_method=args.ensemble_method,
        conformal_alpha=args.conformal_alpha,
        conformal_cal_fraction=args.conformal_cal_fraction,
    )
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
