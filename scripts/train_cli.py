"""Skeleton training CLI for future extension.

Current functionality:
- Placeholder commands for training spectral/lightcurve models.
- Demonstrates where calibration fitting & registry logging would occur.

Extend by implementing actual data loaders, model building, optimization loops.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from stellar_platform.utils.seeding import set_global_seed
from stellar_platform.data.synthetic import generate_synthetic_spectra, generate_synthetic_lightcurves

from stellar_platform.models import spectral as spectral_models
from stellar_platform.models import lightcurve as lc_models
from stellar_platform.models import sed as sed_models

try:  # optional TF import check
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore
from stellar_platform.models.registry import ModelRegistry
from stellar_platform.evaluation import TemperatureScaler
from scripts.evaluate_cli import generate_real_metrics  # reuse evaluation logic


def _dummy_data(n=128, length=256, n_classes=3):
    X = np.random.randn(n, length, 1).astype('float32')
    y = np.random.randint(0, n_classes, size=n)
    return X, y


def train_spectral(args):
    """Train (or synthesize) a spectral model and register it.

    If --force-dummy is supplied or TensorFlow is unavailable, a lightweight JSON
    artifact describing a dummy model is produced instead of a Keras .h5 file.
    """
    # Use synthetic structured spectra for non-dummy path; dummy keeps simple Gaussian noise.
    if getattr(args, 'force_dummy', False) or tf is None:
        X, y = _dummy_data(n=args.samples, length=args.length, n_classes=args.classes)
    else:
        seed_used = set_global_seed()
        print(f"Seed used for synthetic spectral generation: {seed_used}")
        X, y = generate_synthetic_spectra(n=args.samples, length=args.length, n_classes=args.classes, seed=seed_used)
    use_dummy = bool(getattr(args, 'force_dummy', False) or tf is None)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not use_dummy:
        spec = spectral_models.SpectralCNN(input_shape=(args.length, 1), num_classes=args.classes)
        spec.build_model()
        spec.compile_model()
        spec.model.fit(X, np.eye(args.classes)[y], epochs=args.epochs, batch_size=args.batch_size, verbose=0)
        logits = spec.model.predict(X, verbose=0)
        artifact_path = Path(args.output_dir) / "spectral_cnn.keras"
        spec.model.save(str(artifact_path))
        artifacts = {"model": str(artifact_path)}
    else:
        if tf is None and not getattr(args, 'force_dummy', False):
            print("INFO: TensorFlow unavailable; automatically falling back to dummy spectral model.")
        elif getattr(args, 'force_dummy', False):
            print("INFO: Forcing dummy spectral model per --force-dummy.")
        # Random logits to fit calibrator; prediction will emulate softmax later.
        logits = np.random.randn(X.shape[0], args.classes)
        artifact_path = Path(args.output_dir) / "spectral_cnn_dummy.json"
        dummy_desc = {"type": "dummy", "classes": args.classes, "family": "spectral_cnn"}
        artifact_path.write_text(json.dumps(dummy_desc))
        artifacts = {"model": str(artifact_path)}

    scaler = TemperatureScaler().fit(logits, y)
    registry = ModelRegistry()
    version = registry.register(
        "spectral_cnn",
        metrics={"dummy_loss": 0.0},
        artifacts=artifacts,
        params={"num_classes": args.classes, "length": args.length, "dummy": use_dummy},
    )
    meta_path = registry.root / "spectral_cnn" / version / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["calibrator"] = scaler.to_dict()
    if 'seed_used' in locals():
        meta["seed"] = int(seed_used)
    # Auto evaluation headline metrics
    try:
        metrics = generate_real_metrics('spectral_cnn', meta, n_eval=128)
        headline = {k: metrics[k] for k in ('accuracy', 'macro_f1', 'mcc', 'ece') if k in metrics}
        meta['headline_metrics'] = headline
    except Exception as e:
        print(f"WARNING: Auto evaluation failed for spectral_cnn: {e}")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Registered spectral_cnn version {version} (dummy={use_dummy}) with calibration.")


def train_lightcurve(args):
    """Train or synthesize a lightcurve model and register it."""
    if getattr(args, 'force_dummy', False) or tf is None:
        X, y = _dummy_data(n=args.samples, length=args.length, n_classes=args.classes)
    else:
        seed_used = set_global_seed()
        print(f"Seed used for synthetic lightcurve generation: {seed_used}")
        X, y = generate_synthetic_lightcurves(n=args.samples, length=args.length, n_classes=args.classes, seed=seed_used)
    use_dummy = bool(getattr(args, 'force_dummy', False) or tf is None)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not use_dummy:
        lcm = lc_models.LightCurveTransformer(input_shape=(args.length, 1), num_classes=args.classes)
        lcm.build_model()
        lcm.compile_model()
        lcm.model.fit(X, np.eye(args.classes)[y], epochs=args.epochs, batch_size=args.batch_size, verbose=0)
        logits = lcm.model.predict(X, verbose=0)
        artifact_path = Path(args.output_dir) / "lightcurve_transformer.keras"
        lcm.model.save(str(artifact_path))
        artifacts = {"model": str(artifact_path)}
    else:
        if tf is None and not getattr(args, 'force_dummy', False):
            print("INFO: TensorFlow unavailable; automatically falling back to dummy lightcurve model.")
        elif getattr(args, 'force_dummy', False):
            print("INFO: Forcing dummy lightcurve model per --force-dummy.")
        logits = np.random.randn(X.shape[0], args.classes)
        artifact_path = Path(args.output_dir) / "lightcurve_transformer_dummy.json"
        dummy_desc = {"type": "dummy", "classes": args.classes, "family": "lightcurve_transformer"}
        artifact_path.write_text(json.dumps(dummy_desc))
        artifacts = {"model": str(artifact_path)}

    scaler = TemperatureScaler().fit(logits, y)
    registry = ModelRegistry()
    version = registry.register(
        "lightcurve_transformer",
        metrics={"dummy_loss": 0.0},
        artifacts=artifacts,
        params={"num_classes": args.classes, "length": args.length, "dummy": use_dummy},
    )
    meta_path = registry.root / "lightcurve_transformer" / version / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["calibrator"] = scaler.to_dict()
    if 'seed_used' in locals():
        meta["seed"] = int(seed_used)
    try:
        metrics = generate_real_metrics('lightcurve_transformer', meta, n_eval=128)
        headline = {k: metrics[k] for k in ('accuracy', 'macro_f1', 'mcc', 'ece') if k in metrics}
        meta['headline_metrics'] = headline
    except Exception as e:
        print(f"WARNING: Auto evaluation failed for lightcurve_transformer: {e}")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Registered lightcurve_transformer version {version} (dummy={use_dummy}) with calibration.")


def _generate_synthetic_sed(n: int = 256, n_bands: int = 8, n_classes: int = 3, seed: int | None = None):
    rng = np.random.default_rng(seed)
    X = rng.lognormal(mean=0.0, sigma=0.5, size=(n, n_bands))
    # Class influences a couple of band scales
    y = rng.integers(0, n_classes, size=n)
    for i in range(n):
        cls = y[i]
        X[i, cls % n_bands] *= 1.5
        X[i, (cls+1) % n_bands] *= 0.7
    return X.astype('float32'), y


def train_sed(args):
    """Train or synthesize an SED classifier and register it (joblib or dummy JSON)."""
    X, y = _generate_synthetic_sed(n=args.samples, n_bands=args.bands, n_classes=args.classes, seed=42)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    use_dummy = bool(getattr(args, 'force_dummy', False))
    artifacts = {}
    logits_or_probs = None
    if not use_dummy:
        try:
            clf = sed_models.SEDClassifier({'n_estimators': 100})
            clf.build_model(num_classes=args.classes)
            clf.train(X, y)
            # predict_proba for calibration
            probs = clf.predict(X)
            logits_or_probs = probs
            art_path = Path(args.output_dir) / 'sed_model.pkl'
            # Save via joblib inside SEDClassifier.save_model; we want a single path in artifacts
            # For API simplicity, save model pipeline only under a single .pkl
            import joblib  # type: ignore
            joblib.dump(clf, art_path)
            artifacts = {"model": str(art_path)}
        except Exception:
            use_dummy = True

    if use_dummy:
        art_path = Path(args.output_dir) / 'sed_dummy.json'
        desc = {"type": "dummy", "classes": args.classes, "family": "sed"}
        art_path.write_text(json.dumps(desc))
        artifacts = {"model": str(art_path)}
        rng = np.random.default_rng(0)
        probs = rng.random((X.shape[0], args.classes))
        probs /= probs.sum(axis=1, keepdims=True)
        logits_or_probs = probs

    scaler = TemperatureScaler().fit(logits_or_probs, y)
    registry = ModelRegistry()
    version = registry.register(
        "sed",
        metrics={"dummy_loss": 0.0},
        artifacts=artifacts,
        params={"num_classes": args.classes, "bands": args.bands, "dummy": use_dummy},
    )
    meta_path = registry.root / "sed" / version / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["calibrator"] = scaler.to_dict()
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Registered sed version {version} (dummy={use_dummy}) with calibration.")


def main():
    parser = argparse.ArgumentParser(description="Stellar Platform Training CLI (skeleton)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("train-spectral")
    sp.add_argument("--samples", type=int, default=128)
    sp.add_argument("--length", type=int, default=256)
    sp.add_argument("--classes", type=int, default=3)
    sp.add_argument("--output-dir", type=str, default="artifacts")
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--force-dummy", action="store_true", help="Force creation of dummy (non-TF) model artifact")
    sp.set_defaults(func=train_spectral)

    lp = sub.add_parser("train-lightcurve")
    lp.add_argument("--samples", type=int, default=128)
    lp.add_argument("--length", type=int, default=256)
    lp.add_argument("--classes", type=int, default=3)
    lp.add_argument("--output-dir", type=str, default="artifacts")
    lp.add_argument("--epochs", type=int, default=1)
    lp.add_argument("--batch-size", type=int, default=32)
    lp.add_argument("--force-dummy", action="store_true", help="Force creation of dummy (non-TF) model artifact")
    lp.set_defaults(func=train_lightcurve)

    sp_sed = sub.add_parser("train-sed")
    sp_sed.add_argument("--samples", type=int, default=256)
    sp_sed.add_argument("--bands", type=int, default=8)
    sp_sed.add_argument("--classes", type=int, default=3)
    sp_sed.add_argument("--output-dir", type=str, default="artifacts")
    sp_sed.add_argument("--force-dummy", action="store_true", help="Force creation of dummy JSON artifact")
    sp_sed.set_defaults(func=train_sed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
