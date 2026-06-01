#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AROMATIC = set("FWY")
POSITIVE = set("KRH")

def compute_sequence_features(cdr3):
    n = len(cdr3)
    if n == 0:
        return None
    aa_counts = {aa: cdr3.count(aa) for aa in AMINO_ACIDS}
    features = {
        'cdr3_len': n,
        'aromatic_ratio': sum(1 for a in cdr3 if a in AROMATIC) / n,
        'glycine_ratio': aa_counts.get('G', 0) / n,
        'serine_ratio': aa_counts.get('S', 0) / n,
        'proline_count': aa_counts.get('P', 0),
        'positive_count': sum(1 for a in cdr3 if a in POSITIVE),
        'hydrophobic_ratio': sum(1 for a in cdr3 if a in set("AILMFWV")) / n,
        'first_is_aromatic': int(cdr3[0] in AROMATIC),
        'last_is_YH': int(cdr3[-1] in ('Y', 'H')),
        'has_ggg': int('GGG' in cdr3),
        'has_sss': int('SSS' in cdr3),
        'has_ll': int('LL' in cdr3),
    }
    for aa in AMINO_ACIDS:
        features[f'aa_{aa}_ratio'] = aa_counts.get(aa, 0) / n
    for pos in range(min(n, 13)):
        for aa in AMINO_ACIDS:
            features[f'pos{pos}_{aa}'] = int(cdr3[pos] == aa) if pos < n else 0
    return features

def train_scorer(training_csv, feature_matrix_csv):
    feat_df = pd.read_csv(feature_matrix_csv)
    train_df = pd.read_csv(training_csv)

    cdr3_feature_cols = [
        'cdr3_len', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio',
        'proline_count', 'positive_count', 'hydrophobic_ratio',
        'first_is_aromatic', 'last_is_YH', 'has_ggg', 'has_sss', 'has_ll',
    ]
    available_cols = [c for c in cdr3_feature_cols if c in feat_df.columns]

    X = feat_df[available_cols].fillna(0).values
    y = feat_df['rf2_passed'].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"  训练集 AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X_scaled, y)
    return clf, scaler, available_cols

def score_sequences(generated_csv, clf, scaler, feature_cols):
    gen_df = pd.read_csv(generated_csv)
    feature_dict_list = []
    for _, row in gen_df.iterrows():
        cdr3 = row['cdr3']
        feats = compute_sequence_features(cdr3)
        if feats is not None:
            feats['cdr3'] = cdr3
            feature_dict_list.append(feats)

    if not feature_dict_list:
        print("  无有效序列可评分")
        return gen_df

    feat_df = pd.DataFrame(feature_dict_list)
    cdr3_col = feat_df['cdr3']
    feat_df = feat_df.drop(columns=['cdr3'])

    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[feature_cols].fillna(0)

    X = scaler.transform(feat_df.values)
    probas = clf.predict_proba(X)[:, 1]
    result_df = gen_df.copy()
    result_df['predicted_pass_prob'] = probas
    return result_df

def apply_diversity_filter(df, min_edit_distance=3, max_candidates=200):
    df_sorted = df.sort_values('predicted_pass_prob', ascending=False)
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
    parser.add_argument("--prob-threshold", type=float, default=0.3, help="Minimum predicted pass probability")
    args = parser.parse_args()

    print("CSC-O 快速筛选器")

    training_csv = Path(args.training_data)
    feature_matrix_csv = training_csv.parent / "feature_matrix.csv"
    if not feature_matrix_csv.exists():
        feature_matrix_csv = training_csv

    print("  训练评分模型...")
    clf, scaler, feature_cols = train_scorer(str(training_csv), str(feature_matrix_csv))

    print("  评分生成序列...")
    scored_df = score_sequences(args.generated, clf, scaler, feature_cols)
    print(f"  评分完成: {len(scored_df)} 条, 平均预测通过率: {scored_df['predicted_pass_prob'].mean():.3f}")

    filtered = scored_df[scored_df['predicted_pass_prob'] >= args.prob_threshold]
    print(f"  概率阈值过滤 (>= {args.prob_threshold}): {len(filtered)} 条")

    if len(filtered) > args.top_n:
        diverse = apply_diversity_filter(filtered, min_edit_distance=args.min_edit_distance, max_candidates=args.top_n)
        print(f"  多样性过滤 (edit_dist >= {args.min_edit_distance}): {len(diverse)} 条")
    else:
        diverse = filtered

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diverse.to_csv(out_path, index=False)
    print(f"  输出: {out_path} ({len(diverse)} 条)")
    if len(diverse) > 0:
        print(f"  预测通过率范围: [{diverse['predicted_pass_prob'].min():.3f}, {diverse['predicted_pass_prob'].max():.3f}]")
        print(f"  CDR3长度分布: {dict(diverse['cdr3_len'].value_counts().sort_index())}")

if __name__ == "__main__":
    main()
