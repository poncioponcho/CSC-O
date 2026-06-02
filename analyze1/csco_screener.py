#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from csco_config import (
    CDR3_FEATURE_COLS,
    extract_cdr3_features,
    get_optimization_target,
    DEFAULT_CONFIG,
)


def train_scorer(feature_matrix_csv, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    feat_df = pd.read_csv(feature_matrix_csv)

    available_cols = [c for c in CDR3_FEATURE_COLS if c in feat_df.columns]

    X = feat_df[available_cols].fillna(0).values
    y_binary, y_continuous, target_label = get_optimization_target(feat_df, config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    cv_scores = cross_val_score(clf, X_scaled, y_binary, cv=5, scoring='roc_auc')
    print(f"  训练集 AUC ({target_label}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X_scaled, y_binary)
    return clf, scaler, available_cols, target_label


def score_sequences(generated_csv, clf, scaler, feature_cols):
    gen_df = pd.read_csv(generated_csv)
    feature_dict_list = []
    for _, row in gen_df.iterrows():
        cdr3 = row['cdr3']
        feats = extract_cdr3_features(cdr3)
        if feats['cdr3_len'] > 0:
            feats['cdr3'] = cdr3
            feature_dict_list.append(feats)

    if not feature_dict_list:
        print("  无有效序列可评分")
        return gen_df

    feat_df = pd.DataFrame(feature_dict_list)
    feat_df = feat_df.drop(columns=['cdr3'])

    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[feature_cols].fillna(0)

    X = scaler.transform(feat_df.values)
    probas = clf.predict_proba(X)[:, 1]
    result_df = gen_df.copy()
    result_df['predicted_prob'] = probas
    return result_df


def apply_diversity_filter(df, prob_col='predicted_prob', min_edit_distance=3, max_candidates=200):
    df_sorted = df.sort_values(prob_col, ascending=False)
    selected = []
    selected_cdr3s = []

    for _, row in df_sorted.iterrows():
        cdr3 = row['cdr3']
        if len(selected) >= max_candidates:
            break
        is_diverse = True
        for existing in selected_cdr3s:
            if len(cdr3) == len(existing):
                dist = sum(a != b for a, b in zip(cdr3, existing))
                if dist < min_edit_distance:
                    is_diverse = False
                    break
        if is_diverse:
            selected.append(row)
            selected_cdr3s.append(cdr3)

    return pd.DataFrame(selected)


def main():
    parser = argparse.ArgumentParser(description="CSC-O Fast Sequence Screener")
    parser.add_argument("--generated", required=True, help="Generated sequences CSV")
    parser.add_argument("--training-data", required=True, help="Training feature_matrix.csv")
    parser.add_argument("--output", default="./output/screened_candidates.csv", help="Output CSV")
    parser.add_argument("--top-n", type=int, default=200, help="Top N candidates to output")
    parser.add_argument("--min-edit-distance", type=int, default=3, help="Minimum edit distance between candidates")
    parser.add_argument("--prob-threshold", type=float, default=0.3, help="Minimum predicted probability")
    parser.add_argument("--target", choices=["final_candidate", "rf2_passed"], default="final_candidate",
                        help="Optimization target for training")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config['optimization_target'] = args.target

    print("CSC-O 快速筛选器")

    training_path = Path(args.training_data)
    feature_matrix_csv = training_path
    sibling_fm = training_path.parent / "feature_matrix.csv"
    if sibling_fm.exists():
        feature_matrix_csv = sibling_fm

    print("  训练评分模型...")
    clf, scaler, feature_cols, target_label = train_scorer(str(feature_matrix_csv), config)

    print("  评分生成序列...")
    scored_df = score_sequences(args.generated, clf, scaler, feature_cols)
    prob_col = 'predicted_prob'
    print(f"  评分完成: {len(scored_df)} 条, 平均预测概率({target_label}): {scored_df[prob_col].mean():.3f}")

    filtered = scored_df[scored_df[prob_col] >= args.prob_threshold]
    print(f"  概率阈值过滤 (>= {args.prob_threshold}): {len(filtered)} 条")

    if len(filtered) > args.top_n:
        diverse = apply_diversity_filter(filtered, prob_col=prob_col,
                                         min_edit_distance=args.min_edit_distance,
                                         max_candidates=args.top_n)
        print(f"  多样性过滤 (edit_dist >= {args.min_edit_distance}): {len(diverse)} 条")
    else:
        diverse = filtered

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diverse.to_csv(out_path, index=False)
    print(f"  输出: {out_path} ({len(diverse)} 条)")
    if len(diverse) > 0:
        print(f"  预测概率范围: [{diverse[prob_col].min():.3f}, {diverse[prob_col].max():.3f}]")
        if 'cdr3_len' in diverse.columns:
            print(f"  CDR3长度分布: {dict(diverse['cdr3_len'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()
