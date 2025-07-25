#!/usr/bin/env python3
"""
Benchmark Suite: Imbalanced Financial Fraud Detection Engine (ML Project 7)
Evaluates Cost-Sensitive HistGradientBoosting, Balanced Random Forest, and
Majority-Undersampled Logistic Regression (Notebook Baseline) on skewed PCA
credit card transaction streams (30 features: Time, V1..V28, Amount).
"""

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler


def generate_credit_card_stream(n_samples: int = 6000, random_state: int = 42):
    """
    Synthesizes a 30-dimensional credit card transaction dataset (Time, V1..V28 PCA
    components, Amount) with severe class imbalance (3.0% fraud prevalence).
    """
    rng = np.random.default_rng(random_state)
    X_pca, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=18,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.97, 0.03],
        class_sep=1.45,
        flip_y=0.008,
        random_state=random_state,
    )
    time_col = np.sort(rng.uniform(0, 172792, size=(n_samples, 1)))
    amount_col = np.where(
        y == 1,
        rng.lognormal(mean=4.1, sigma=1.35, size=n_samples),
        rng.lognormal(mean=3.6, sigma=1.15, size=n_samples),
    ).reshape(-1, 1)

    X = np.hstack([time_col, X_pca, amount_col])
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return X, y, feature_names


def render_svg_chart(results: list[dict], output_path: Path) -> None:
    """Generates a dark-theme vector SVG chart of imbalanced fraud detection metrics."""
    width, height = 880, 440
    metrics = ["roc_auc", "pr_auc", "f1", "recall"]
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC (Avg Precision)",
        "f1": "Fraud F1-Score",
        "recall": "Fraud Recall (Sensitivity)",
    }
    colors = {
        "roc_auc": "#38bdf8",
        "pr_auc": "#34d399",
        "f1": "#a78bfa",
        "recall": "#fbbf24",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        "  <defs>",
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0f172a"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        "Imbalanced Credit Card Fraud Detection — 5-Fold Stratified CV</text>",
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        "N = 6,000 Transactions (30 Features: Time, V1–V28, Amount) | 3.0% Fraud Prevalence</text>",
    ]

    lx = 32
    for m in metrics:
        svg.append(f'  <rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(
            f'  <text x="{lx + 18}" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" '
            f'font-size="12" font-weight="600">{metric_labels[m]}</text>'
        )
        lx += 205

    chart_left = 295
    chart_right = 810
    chart_width = chart_right - chart_left
    top_y = 125
    group_height = 94

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 10}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 10}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 32}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="12.5" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 50}" fill="#64748b" '
            f'font-family="Consolas, monospace" font-size="11" text-anchor="end">Precision: {entry["precision"]:.4f} | Acc: {entry["accuracy"]:.4f}</text>'
        )
        for m_idx, m in enumerate(metrics):
            val = entry[m]
            bar_w = max(2.0, val * chart_width)
            by = gy + m_idx * 18
            svg.append(
                f'  <rect x="{chart_left}" y="{by}" width="{bar_w:.1f}" height="14" rx="3" fill="{colors[m]}" opacity="0.92"/>'
            )
            svg.append(
                f'  <text x="{chart_left + bar_w + 8:.1f}" y="{by + 11}" fill="#e2e8f0" '
                f'font-family="Consolas, monospace" font-size="11" font-weight="600">{val:.4f}</text>'
            )

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    X, y, feature_names = generate_credit_card_stream()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rng = np.random.default_rng(42)

    configs = [
        ("Cost-Sensitive HistGradientBoosting", "hgb"),
        ("Balanced Subsample Random Forest", "rf"),
        ("Undersampled Logistic Regression", "undersampled_lr"),
    ]

    results = []
    for label, kind in configs:
        oof_probs = np.zeros(len(y), dtype=np.float64)
        oof_preds = np.zeros(len(y), dtype=int)

        for train_idx, val_idx in cv.split(X, y):
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            y_tr = y[train_idx]
            X_va = scaler.transform(X[val_idx])

            if kind == "hgb":
                clf = HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.06,
                    max_depth=6,
                    class_weight="balanced",
                    random_state=42,
                )
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.45).astype(int)
            elif kind == "rf":
                clf = RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    class_weight="balanced_subsample",
                    random_state=42,
                )
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.32).astype(int)
            else:
                # Replicate notebook's 1:1 majority undersampling on training split
                pos_idx = np.where(y_tr == 1)[0]
                neg_idx = np.where(y_tr == 0)[0]
                sampled_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
                bal_idx = np.concatenate([pos_idx, sampled_neg])
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                clf.fit(X_tr[bal_idx], y_tr[bal_idx])
                probs = clf.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.50).astype(int)

            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds

        results.append(
            {
                "model": label,
                "roc_auc": round(float(roc_auc_score(y, oof_probs)), 4),
                "pr_auc": round(float(average_precision_score(y, oof_probs)), 4),
                "f1": round(float(f1_score(y, oof_preds)), 4),
                "precision": round(float(precision_score(y, oof_preds)), 4),
                "recall": round(float(recall_score(y, oof_preds)), 4),
                "accuracy": round(float(accuracy_score(y, oof_preds)), 4),
            }
        )

    results.sort(key=lambda r: (r["pr_auc"], r["roc_auc"]), reverse=True)

    root_dir = Path(__file__).resolve().parent.parent
    json_path = root_dir / "benchmarks" / "benchmark_results.json"
    svg_path = root_dir / "assets" / "benchmark_chart.svg"

    payload = {
        "project": "Imbalanced Financial Fraud Detection Engine (ML Project 7)",
        "dataset": "PCA Credit Card Transaction Stream (N=6000, d=30, Fraud Rate=3.2%)",
        "validation_protocol": "5-Fold Stratified Cross-Validation on Unaltered Imbalanced Folds",
        "features": feature_names,
        "results": results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(results, svg_path)
    print(f"[OK] Saved fraud benchmark JSON -> {json_path}")
    print(f"[OK] Saved fraud benchmark SVG  -> {svg_path}")


if __name__ == "__main__":
    main()
