"""End-to-end baseline: load -> features -> subject-wise CV -> results/.

Real data:       python scripts/run_baseline.py --subjects 0 1 2 3 4 5 6 7 8 9
Pipeline check:  python scripts/run_baseline.py --synthetic
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import cross_validate, metrics, plot_confusion, plot_leakage
from src.features import add_context, extract_features
from src.models import get_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic data (NOT real EEG) to test the pipeline")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--data-path", default=None, help="MNE data cache folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.synthetic:
        from src.synthetic import make_dataset
        X, y, groups, sfreq = make_dataset(seed=args.seed)
        tag = "synthetic"
    else:
        from src.data import load_dataset
        X, y, groups, sfreq = load_dataset(args.subjects, args.data_path)
        tag = "sleep-edf"
    print(f"{len(y)} epochs, {len(np.unique(groups))} subjects, class counts {np.bincount(y)}")

    F = extract_features(X, sfreq)
    Fc = add_context(F, groups)

    results = {"data": tag, "n_epochs": int(len(y)), "n_subjects": int(len(np.unique(groups)))}
    models = get_models(args.seed)
    for name, model in models.items():
        for feats_name, feats in [("single_epoch", F), ("with_context", Fc)]:
            pred = cross_validate(model, feats, y, groups, "subject", seed=args.seed)
            m = metrics(y, pred)
            results[f"{name}/{feats_name}/subject_wise"] = m
            print(f"{name:24s} {feats_name:13s} subject-wise  "
                  f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} kappa={m['kappa']:.3f}")

    # Leakage demonstration with the main model
    rf = models["random_forest"]
    p_sub = cross_validate(rf, Fc, y, groups, "subject", seed=args.seed)
    p_rnd = cross_validate(rf, Fc, y, groups, "random", seed=args.seed)
    m_sub, m_rnd = metrics(y, p_sub), metrics(y, p_rnd)
    results["leakage_demo"] = {"subject_wise": m_sub, "random_split": m_rnd}
    print(f"leakage demo  subject-wise acc={m_sub['accuracy']:.3f}  "
          f"random-split acc={m_rnd['accuracy']:.3f}")

    plot_confusion(y, p_sub, os.path.join(args.out, f"confusion_{tag}.png"),
                   title=f"RF + context, subject-wise ({tag})")
    plot_leakage(m_sub, m_rnd, os.path.join(args.out, f"leakage_{tag}.png"))
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {args.out}/")


if __name__ == "__main__":
    main()
