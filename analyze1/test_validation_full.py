#!/usr/bin/env python3
"""分段测试验证框架"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('output_v3_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Step 1: 基础验证框架测试 ===
print("=== Step 1: 验证框架基础测试 ===")
from csco_validation_framework import ValidationFramework, ExperimentResult
from csco_funnel_aware_strategy import FunnelAwareStrategy

strategy = FunnelAwareStrategy(verbose=False)
vf = ValidationFramework(strategy=strategy, verbose=True)

# === Step 2: 模拟导入实验数据 ===
print("\n=== Step 2: 模拟导入实验数据 ===")
np.random.seed(42)

# 加载生成序列
gen_csv = 'output_v3_funnel/v30_generated_scored.csv'
if Path(gen_csv).exists():
    gen_df = pd.read_csv(gen_csv)
    seq_col = 'cdr3' if 'cdr3' in gen_df.columns else 'sequence'
    if seq_col not in gen_df.columns:
        seq_col = gen_df.columns[0]
    sequences = gen_df[seq_col].dropna().tolist()[:200]
else:
    # 使用简单序列
    sequences = []
    first_aas = ['F', 'W', 'Y', 'G', 'V', 'A', 'D', 'T']
    last_aas = ['Y', 'A', 'H', 'N', 'D']
    import random
    rng = random.Random(42)
    for _ in range(200):
        l = rng.choice([6, 7])
        first = rng.choice(first_aas)
        last = rng.choice(last_aas)
        middle = ''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY')) for _ in range(l - 2))
        sequences.append(first + middle + last)

sim_results = []
for i, seq in enumerate(sequences):
    # 基于漏斗评分模拟RF2/FC结果
    try:
        r = strategy.score_sequence_funnel(seq)
        p_rf2 = 0.15 + 0.03 * r.rf2_score  # 简化映射
        p_fc_given_rf2 = 0.02 + 0.005 * r.final_score
    except Exception:
        p_rf2 = 0.15
        p_fc_given_rf2 = 0.02

    p_rf2 = np.clip(p_rf2, 0.01, 0.95)
    p_fc_given_rf2 = np.clip(p_fc_given_rf2, 0.001, 0.5)

    rf2_passed = bool(np.random.random() < p_rf2)
    fc = bool(np.random.random() < p_fc_given_rf2) if rf2_passed else False

    sim_results.append(ExperimentResult(
        sequence=seq, rf2_passed=rf2_passed, final_candidate=fc,
        notes=f"simulated #{i}"
    ))

ingest_stats = vf.ingest_experimental_results(sim_results)

sim_rf2_rate = sum(1 for r in sim_results if r.rf2_passed) / len(sim_results)
sim_fc_rate = sum(1 for r in sim_results if r.final_candidate) / len(sim_results)
sim_fc_given_rf2 = sum(1 for r in sim_results if r.final_candidate) / max(
    sum(1 for r in sim_results if r.rf2_passed), 1)

print(f"\n模拟实验统计:")
print(f"  总数: {len(sim_results)}")
print(f"  RF2通过率: {sim_rf2_rate:.2%}")
print(f"  FC率: {sim_fc_rate:.2%}")
print(f"  P(FC|RF2): {sim_fc_given_rf2:.2%}")

# === Step 3: 贝叶斯更新 ===
print("\n=== Step 3: 贝叶斯更新 ===")
# 记录先验
prior_records = []
for param, vals in vf.priors.items():
    prior_records.append({'parameter': param, 'prior_mean': vals['mean'], 'prior_std': vals['std']})

updates = vf.bayesian_update()

# 记录后验
posterior_records = []
for param, vals in vf.posteriors.items():
    posterior_records.append({'parameter': param, 'posterior_mean': vals['mean'], 'posterior_std': vals['std']})

compare_df = pd.DataFrame(prior_records).merge(pd.DataFrame(posterior_records), on='parameter')
compare_df['delta_mean'] = compare_df['posterior_mean'] - compare_df['prior_mean']
compare_df['pct_change'] = (compare_df['delta_mean'] / compare_df['prior_mean'].abs() * 100).round(2)
kl_map = {u.parameter: u.kl_divergence for u in updates}
compare_df['kl_divergence'] = compare_df['parameter'].map(kl_map).fillna(0)

compare_df.to_csv(OUTPUT_DIR / 'bayesian_prior_posterior_comparison.csv', index=False)

print("\n先验→后验参数变化:")
print(compare_df.to_string(index=False))

# === Step 4: AUC评估 ===
print("\n=== Step 4: AUC评估 ===")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df_orig = pd.read_csv('output_server_v2.3/feature_matrix.csv')
scores = []
for _, row in df_orig.iterrows():
    seq = row.get('cdr3_sequence', '')
    if pd.isna(seq) or len(seq) < 5:
        scores.append(0.0)
        continue
    try:
        r = strategy.score_sequence_funnel(seq)
        scores.append(r.combined_score)
    except Exception:
        scores.append(0.0)

y_true = df_orig['final_candidate'].astype(int).values
y_score = np.array(scores)
X = np.array(scores).reshape(-1, 1)

try:
    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    cv_auc = cross_val_score(lr, X, y_true, cv=5, scoring='roc_auc')
    auc_mean = cv_auc.mean()
    auc_std = cv_auc.std()
except Exception:
    try:
        auc_mean = roc_auc_score(y_true, y_score)
        auc_std = 0.0
    except Exception:
        auc_mean = 0.0
        auc_std = 0.0

print(f"  CV-AUC: {auc_mean:.4f} +/- {auc_std:.4f}")

# === Step 5: HR/ATE汇总 ===
print("\n=== Step 5: HR/ATE汇总 ===")
from csco_multistage_causal import MultiStageMediationModel
from csco_multistate_survival import MultiStateSurvivalModel

emb = np.load('output_server_v2.3/esm2_embeddings.npy')

med_model = MultiStageMediationModel(available_stages=['rf2', 'final'], method='decomposition', verbose=False)
med_results = med_model.fit(
    df=df_orig, embeddings=emb,
    treatment_cols=['first_is_aromatic', 'cdr3_length_bin', 'glycine_ratio_bin', 'serine_ratio_bin'],
    confounder_cols=['backbone_id'],
)
med_model.save_results(OUTPUT_DIR / 'mediation_effects.csv')

surv_model = MultiStateSurvivalModel(available_stages=['rf2', 'final'], verbose=False)
surv_results = surv_model.fit(
    df=df_orig,
    treatment_cols=['first_is_aromatic', 'cdr3_length_bin', 'glycine_ratio_bin', 'serine_ratio_bin'],
    confounder_cols=['backbone_id'],
)
surv_model.save_results(OUTPUT_DIR / 'multistate_hazard_ratios.csv')

cv_df = surv_model.cross_validate_with_mediation(med_results)
cv_df.to_csv(OUTPUT_DIR / 'cross_validation.csv', index=False)

# HR汇总
hr_rows = []
for t, r in surv_results.items():
    for stage, sr in r.stage_results.items():
        hr_rows.append({'treatment': t, 'stage': stage, 'hr': round(sr.hr, 4),
                        'hr_ci': f"[{sr.hr_ci[0]:.3f}, {sr.hr_ci[1]:.3f}]",
                        'p_value': f"{sr.p_value:.4f}"})
    hr_rows.append({'treatment': t, 'stage': 'combined', 'hr': round(r.combined_hr, 4),
                    'hr_ci': f"[{r.combined_hr_ci[0]:.3f}, {r.combined_hr_ci[1]:.3f}]",
                    'p_value': 'N/A'})
hr_df = pd.DataFrame(hr_rows)
hr_df.to_csv(OUTPUT_DIR / 'hr_summary.csv', index=False)
print("\nHR汇总:")
print(hr_df.to_string(index=False))

# ATE汇总
ate_rows = []
for t, r in med_results.items():
    ate_rows.append({'treatment': t, 'total_effect': round(r.total_effect, 4),
                     'direct_effect': round(r.direct_effect, 4),
                     'indirect_effect': round(r.indirect_effect, 4),
                     'total_se': round(r.total_effect_se, 4),
                     'direct_se': round(r.direct_effect_se, 4)})
ate_df = pd.DataFrame(ate_rows)
ate_df.to_csv(OUTPUT_DIR / 'ate_summary.csv', index=False)
print("\nATE汇总:")
print(ate_df.to_string(index=False))

# === Step 6: 生成器日志增强 ===
print("\n=== Step 6: 生成器日志增强 ===")
from csco_funnel_generator import FunnelAwareGenerator

gen = FunnelAwareGenerator(strategy=strategy, min_edit_distance=3, verbose=True)
gen_seqs = gen.generate(n_samples=2000, top_n=100, seed=42)

log_rows = []
for i, s in enumerate(gen_seqs):
    log_rows.append({
        'step': i + 1,
        'sequence': s.cdr3,
        'p_final': s.estimated_p_final,
        'p_rf2': s.estimated_p_rf2,
        'p_final_given_rf2': s.estimated_p_final_given_rf2,
        'combined_score': s.combined_score,
        'diversity_score': s.diversity_score,
        'first_aa': s.first_aa,
    })
    if (i + 1) % 20 == 0:
        avg_pf = np.mean([r['p_final'] for r in log_rows])
        avg_div = np.mean([r['diversity_score'] for r in log_rows])
        print(f"  Step {i+1:3d}: P(final)_avg={avg_pf:.6f}, diversity_avg={avg_div:.2f}")

log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUTPUT_DIR / 'generation_step_log.csv', index=False)

# Shannon熵
from collections import Counter
seq_list = [s.cdr3 for s in gen_seqs]
seq_counts = Counter(seq_list)
total = len(seq_list)
probs = [c / total for c in seq_counts.values()]
shannon = -sum(p * np.log2(p) for p in probs if p > 0)

# === 最终报告 ===
print("\n" + "=" * 70)
print("CSC-O v3.1 验证框架测试报告")
print("=" * 70)

report = {
    'version': 'v3.1',
    'auc': {'cv_auc_mean': round(auc_mean, 4), 'cv_auc_std': round(auc_std, 4), 'target': 0.80},
    'hr': hr_rows,
    'ate': ate_rows,
    'generation': {
        'n_sequences': len(gen_seqs),
        'avg_p_final': round(np.mean([s.estimated_p_final for s in gen_seqs]), 6),
        'shannon_entropy': round(shannon, 4),
        'diversity_avg': round(np.mean([s.diversity_score for s in gen_seqs]), 2),
    },
    'bayesian': {
        'n_updated': len(updates),
        'updates': [u.to_dict() for u in updates],
    },
    'simulated': {
        'n': len(sim_results),
        'rf2_rate': round(sim_rf2_rate, 4),
        'fc_rate': round(sim_fc_rate, 4),
        'fc_given_rf2': round(sim_fc_given_rf2, 4),
    },
    'cross_validation': {
        'consistent': int(cv_df['direction_consistent'].sum()),
        'total': len(cv_df),
    },
}

import json
with open(OUTPUT_DIR / 'validation_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"""
AUC: {auc_mean:.4f} +/- {auc_std:.4f} (target >= 0.80)
HR: 见hr_summary.csv
ATE: 见ate_summary.csv
生成: {len(gen_seqs)} seqs, P(final)={np.mean([s.estimated_p_final for s in gen_seqs]):.6f}, Shannon={shannon:.4f}
贝叶斯: {len(updates)} params updated
模拟: RF2={sim_rf2_rate:.2%}, FC={sim_fc_rate:.2%}
交叉验证: {int(cv_df['direction_consistent'].sum())}/{len(cv_df)} consistent
输出目录: {OUTPUT_DIR}/
""")
