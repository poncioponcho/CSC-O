#!/usr/bin/env python3
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)
from csco_config import CDR3_FEATURE_COLS, extract_cdr3_features, get_optimization_target, DEFAULT_CONFIG


def evaluate_model(X, y, target_name, n_splits=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {'target': target_name, 'n_samples': len(y), 'n_positive': int(y.sum()),
               'positive_rate': float(y.mean()), 'n_features': X_scaled.shape[1]}

    t0 = time.time()
    auc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='roc_auc')
    ap_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='average_precision')
    acc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='f1')
    prec_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='precision')
    rec_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='recall')
    elapsed = time.time() - t0

    metrics.update({
        'auc': round(float(auc_scores.mean()), 4), 'auc_std': round(float(auc_scores.std()), 4),
        'ap': round(float(ap_scores.mean()), 4), 'ap_std': round(float(ap_scores.std()), 4),
        'accuracy': round(float(acc_scores.mean()), 4), 'accuracy_std': round(float(acc_scores.std()), 4),
        'f1': round(float(f1_scores.mean()), 4), 'f1_std': round(float(f1_scores.std()), 4),
        'precision': round(float(prec_scores.mean()), 4), 'precision_std': round(float(prec_scores.std()), 4),
        'recall': round(float(rec_scores.mean()), 4), 'recall_std': round(float(rec_scores.std()), 4),
        'cv_time_sec': round(elapsed, 2),
    })

    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

    fi = clf.feature_importances_
    top_features = sorted(zip(X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])], fi),
                          key=lambda x: -x[1])[:10]
    metrics['top_features'] = [(f, round(float(i), 4)) for f, i in top_features]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    metrics['threshold_analysis'] = []
    for t in thresholds:
        y_t = (y_prob >= t).astype(int)
        n_selected = int(y_t.sum())
        if n_selected > 0:
            precision_t = float(((y_t == 1) & (y == 1)).sum()) / n_selected
        else:
            precision_t = 0.0
        metrics['threshold_analysis'].append({
            'threshold': t, 'n_selected': n_selected, 'precision': round(precision_t, 4),
        })

    return metrics


def main():
    data_dir = Path("/home/test/CSC-O/output_v5")
    feat_path = data_dir / "feature_matrix.csv"
    if not feat_path.exists():
        data_dir = Path("/home/test/CSC-O/output")
        feat_path = data_dir / "feature_matrix.csv"
    if not feat_path.exists():
        print("ERROR: feature_matrix.csv not found")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    available_cols = [c for c in CDR3_FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0)

    print("=" * 70)
    print("CSC-O Final Candidate vs RF2 Passed - Performance Evaluation")
    print(f"Data: {len(df)} samples, {len(available_cols)} features")
    print("=" * 70)

    y_fc = df['final_candidate'].astype(int).values
    y_rf2 = df['rf2_passed'].astype(int).values

    print(f"\n--- Target Distribution ---")
    print(f"  final_candidate: {y_fc.sum()}/{len(y_fc)} = {y_fc.mean():.4f} ({y_fc.mean()*100:.2f}%)")
    print(f"  rf2_passed:      {y_rf2.sum()}/{len(y_rf2)} = {y_rf2.mean():.4f} ({y_rf2.mean()*100:.2f}%)")

    print(f"\n{'='*70}")
    print("PART 1: Model Performance Comparison")
    print(f"{'='*70}")

    print(f"\n--- Evaluating rf2_passed model ---")
    m_rf2 = evaluate_model(X, y_rf2, 'rf2_passed')

    print(f"\n--- Evaluating final_candidate model ---")
    m_fc = evaluate_model(X, y_fc, 'final_candidate')

    print(f"\n{'Metric':<20s} {'rf2_passed':>15s} {'final_candidate':>15s} {'Delta':>12s}")
    print("-" * 65)
    for key in ['auc', 'ap', 'accuracy', 'f1', 'precision', 'recall', 'cv_time_sec']:
        v1 = m_rf2[key]; v2 = m_fc[key]
        delta = v2 - v1
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {key:<18s} {v1:>15.4f} {v2:>15.4f} {delta:>+10.4f} {arrow}")

    print(f"\n{'Metric':<20s} {'rf2_passed':>15s} {'final_candidate':>15s}")
    print("-" * 55)
    print(f"  {'n_positive':<18s} {m_rf2['n_positive']:>15d} {m_fc['n_positive']:>15d}")
    print(f"  {'positive_rate':<18s} {m_rf2['positive_rate']:>15.4f} {m_fc['positive_rate']:>15.4f}")
    print(f"  {'TP':<18s} {m_rf2['tp']:>15d} {m_fc['tp']:>15d}")
    print(f"  {'FP':<18s} {m_rf2['fp']:>15d} {m_fc['fp']:>15d}")
    print(f"  {'FN':<18s} {m_rf2['fn']:>15d} {m_fc['fn']:>15d}")
    print(f"  {'TN':<18s} {m_rf2['tn']:>15d} {m_fc['tn']:>15d}")

    print(f"\n--- Top 10 Features (rf2_passed) ---")
    for f, i in m_rf2['top_features']:
        print(f"  {f:<25s} {i:.4f}")

    print(f"\n--- Top 10 Features (final_candidate) ---")
    for f, i in m_fc['top_features']:
        print(f"  {f:<25s} {i:.4f}")

    print(f"\n--- Threshold Analysis (final_candidate) ---")
    print(f"  {'Threshold':>10s} {'N Selected':>12s} {'Precision':>10s}")
    for t in m_fc['threshold_analysis']:
        print(f"  {t['threshold']:>10.1f} {t['n_selected']:>12d} {t['precision']:>10.4f}")

    print(f"\n{'='*70}")
    print("PART 2: Funnel Analysis by CDR3 Length")
    print(f"{'='*70}")

    print(f"\n{'Len':>4s} {'N':>6s} {'RF2 Pass':>9s} {'FC Rate':>8s} {'FC|RF2':>8s} {'FC/Total':>9s}")
    print("-" * 50)
    for length in sorted(df['cdr3_len'].unique()):
        sub = df[df['cdr3_len'] == length]
        n = len(sub)
        rf2_rate = sub['rf2_passed'].mean()
        fc_rate = sub['final_candidate'].mean()
        rf2_pass = sub[sub['rf2_passed'] == 1]
        fc_given_rf2 = rf2_pass['final_candidate'].mean() if len(rf2_pass) > 0 else 0
        print(f"  {int(length):>4d} {n:>6d} {rf2_rate:>9.3f} {fc_rate:>8.4f} {fc_given_rf2:>8.4f} {int(sub['final_candidate'].sum()):>6d}/{n}")

    print(f"\n{'='*70}")
    print("PART 3: Feature Importance Shift (rf2_passed vs final_candidate)")
    print(f"{'='*70}")

    rf2_fi = dict(m_rf2['top_features'])
    fc_fi = dict(m_fc['top_features'])
    all_features = set(list(rf2_fi.keys()) + list(fc_fi.keys()))
    print(f"\n  {'Feature':<25s} {'rf2_imp':>10s} {'fc_imp':>10s} {'Shift':>10s}")
    print("-" * 58)
    for f in sorted(all_features, key=lambda x: abs(fc_fi.get(x, 0) - rf2_fi.get(x, 0)), reverse=True):
        r = rf2_fi.get(f, 0.0)
        c = fc_fi.get(f, 0.0)
        shift = c - r
        arrow = "↑" if shift > 0.01 else "↓" if shift < -0.01 else "="
        print(f"  {f:<25s} {r:>10.4f} {c:>10.4f} {shift:>+10.4f} {arrow}")

    print(f"\n{'='*70}")
    print("PART 4: Generated Sequence Quality Assessment")
    print(f"{'='*70}")

    gen_path = data_dir / "generated_sequences.csv"
    screen_path = data_dir / "screened_candidates.csv"

    if gen_path.exists():
        gen_df = pd.read_csv(gen_path)
        print(f"\n  Generated: {len(gen_df)} sequences")
        print(f"  Length distribution: {dict(gen_df['cdr3_len'].value_counts().sort_index())}")
        if 'soft_score' in gen_df.columns:
            print(f"  Soft score: mean={gen_df['soft_score'].mean():.2f}, "
                  f"median={gen_df['soft_score'].median():.2f}")

    if screen_path.exists():
        scr_df = pd.read_csv(screen_path)
        print(f"\n  Screened: {len(scr_df)} candidates")
        print(f"  Length distribution: {dict(scr_df['cdr3_len'].value_counts().sort_index())}")
        if 'predicted_prob' in scr_df.columns:
            print(f"  Predicted prob: mean={scr_df['predicted_prob'].mean():.3f}, "
                  f"min={scr_df['predicted_prob'].min():.3f}, max={scr_df['predicted_prob'].max():.3f}")
            for length in sorted(scr_df['cdr3_len'].unique()):
                sub = scr_df[scr_df['cdr3_len'] == length]
                print(f"    Length {int(length)}: {len(sub)} seqs, "
                      f"prob={sub['predicted_prob'].mean():.3f}")

    results = {
        'rf2_passed_model': {k: v for k, v in m_rf2.items() if not isinstance(v, list) or k == 'top_features'},
        'final_candidate_model': {k: v for k, v in m_fc.items() if not isinstance(v, list) or k == 'top_features'},
    }
    out_json = data_dir / "model_evaluation_results.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
