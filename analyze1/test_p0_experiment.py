#!/usr/bin/env python3
"""
P0实验: v2.4 vs v3.0 生成序列对比

步骤:
1. 用v2.4策略生成10K序列
2. 用v3.0策略生成10K序列
3. 用漏斗感知模型评分两组序列
4. 对比P(RF2), P(final|RF2), P(final)分布
5. 对比FC正样本覆盖率
"""
import pandas as pd
import numpy as np
import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csco_generator import generate_cdr3, check_hard_constraints, score_soft_preferences, load_strategy
from csco_funnel_aware_strategy import FunnelAwareStrategy
from csco_config import extract_cdr3_features, AROMATIC

output_dir = 'output_v3_funnel'
os.makedirs(output_dir, exist_ok=True)

# === 加载策略 ===
v24_strategy = load_strategy('output_v2.4_test/design_strategy_v2.4.json')
v30_strategy = load_strategy(f'{output_dir}/design_strategy_v3.0.json')

print("="*70)
print("P0实验: v2.4 vs v3.1 (含G条件约束) 生成序列对比")
print("="*70)

# === 生成序列 ===
import random
rng = random.Random(42)
np.random.seed(42)

N_SAMPLES = 10000

def generate_batch(strategy, label, rng, n=10000):
    """用给定策略生成一批序列"""
    hc = strategy['hard_constraints']
    allowed_lengths = hc.get('cdr3_length_allowed', [6, 7])
    
    all_seqs = []
    for length in allowed_lengths:
        n_per = n // len(allowed_lengths)
        seqs = generate_cdr3(strategy, length, n_per, rng, verbose=False)
        all_seqs.extend(seqs)
    
    # 去重
    seen = set()
    unique = []
    for s in all_seqs:
        if s['cdr3'] not in seen:
            seen.add(s['cdr3'])
            unique.append(s)
    
    return unique

print(f"\n--- 生成v2.4序列 ---")
rng24 = random.Random(42)
np.random.seed(42)
v24_seqs = generate_batch(v24_strategy, 'v2.4', rng24, N_SAMPLES)
print(f"  v2.4生成: {len(v24_seqs)} 条")

print(f"\n--- 生成v3.0序列 ---")
rng30 = random.Random(42)
np.random.seed(42)
v30_seqs = generate_batch(v30_strategy, 'v3.0', rng30, N_SAMPLES)
print(f"  v3.0生成: {len(v30_seqs)} 条")

# === 漏斗感知评分 ===
print(f"\n--- 漏斗感知评分 ---")
fas = FunnelAwareStrategy(verbose=False)

def score_batch(seqs, label):
    """对一批序列进行漏斗感知评分"""
    results = []
    for s in seqs:
        cdr3 = s['cdr3']
        funnel_result = fas.score_sequence_funnel(cdr3)
        results.append({
            'cdr3': cdr3,
            'length': len(cdr3),
            'first_aa': cdr3[0],
            'first_is_aromatic': cdr3[0] in AROMATIC,
            'rf2_score': funnel_result.rf2_score,
            'final_score': funnel_result.final_score,
            'combined_score': funnel_result.combined_score,
            'soft_score': s.get('soft_score', 0),
        })
    return pd.DataFrame(results)

v24_df = score_batch(v24_seqs, 'v2.4')
v30_df = score_batch(v30_seqs, 'v3.0')

# === 对比统计 ===
print(f"\n{'='*70}")
print(f"对比结果: v2.4 (n={len(v24_df)}) vs v3.0 (n={len(v30_df)})")
print(f"{'='*70}")

# 1. 首残基分布
print(f"\n--- 1. 首残基分布 ---")
v24_first = v24_df['first_aa'].value_counts()
v30_first = v30_df['first_aa'].value_counts()
all_first_aas = sorted(set(v24_first.index) | set(v30_first.index))
print(f"{'AA':<4} {'v2.4':>8} {'v3.0':>8} {'差异':>8}")
for aa in all_first_aas:
    c24 = v24_first.get(aa, 0)
    c30 = v30_first.get(aa, 0)
    tag = "芳香族" if aa in AROMATIC else "非芳香族"
    print(f"{aa:<4} {c24:>8} {c30:>8} {c30-c24:>+8}  ({tag})")

# 2. 芳香族首残基率
v24_aromatic_rate = v24_df['first_is_aromatic'].mean()
v30_aromatic_rate = v30_df['first_is_aromatic'].mean()
print(f"\n芳香族首残基率: v2.4={v24_aromatic_rate:.1%}, v3.0={v30_aromatic_rate:.1%}")

# 3. 漏斗感知评分
print(f"\n--- 2. 漏斗感知评分 ---")
for metric in ['rf2_score', 'final_score', 'combined_score']:
    v24_mean = v24_df[metric].mean()
    v30_mean = v30_df[metric].mean()
    v24_std = v24_df[metric].std()
    v30_std = v30_df[metric].std()
    print(f"  {metric}: v2.4={v24_mean:.3f}±{v24_std:.3f}, v3.0={v30_mean:.3f}±{v30_std:.3f}, Δ={v30_mean-v24_mean:+.3f}")

# 4. 估计P(final) = P(RF2) × P(final|RF2)
# 用原始数据的基线率作为参考
df_orig = pd.read_csv('output_server_v2.3/feature_matrix.csv')
p_rf2_baseline = df_orig['rf2_passed'].mean()
p_final_given_rf2_baseline = df_orig[df_orig['rf2_passed']==True]['final_candidate'].mean() if df_orig['rf2_passed'].sum() > 0 else 0
p_final_baseline = p_rf2_baseline * p_final_given_rf2_baseline

print(f"\n--- 3. 估算P(final) ---")
print(f"  基线: P(RF2)={p_rf2_baseline:.4f}, P(final|RF2)={p_final_given_rf2_baseline:.4f}, P(final)={p_final_baseline:.6f}")

# 用漏斗评分估算P(final)
# 方法: combined_score > 0 的序列比例 × 基线P(final)
# 更精确: 用评分分位数映射到P(final)
v24_combined_pos = (v24_df['combined_score'] > 0).mean()
v30_combined_pos = (v30_df['combined_score'] > 0).mean()

# 按首残基分组估算
print(f"\n--- 4. 按首残基类型分组估算 ---")
for label, df in [('v2.4', v24_df), ('v3.0', v30_df)]:
    aromatic = df[df['first_is_aromatic']==True]
    non_aromatic = df[df['first_is_aromatic']==False]
    
    print(f"\n  {label}:")
    print(f"    芳香族首残基 (n={len(aromatic)}): "
          f"RF2_score={aromatic['rf2_score'].mean():.3f}, "
          f"Final_score={aromatic['final_score'].mean():.3f}, "
          f"Combined={aromatic['combined_score'].mean():.3f}")
    if len(non_aromatic) > 0:
        print(f"    非芳香族首残基 (n={len(non_aromatic)}): "
              f"RF2_score={non_aromatic['rf2_score'].mean():.3f}, "
              f"Final_score={non_aromatic['final_score'].mean():.3f}, "
              f"Combined={non_aromatic['combined_score'].mean():.3f}")

# 5. FC正样本覆盖率
print(f"\n--- 5. FC正样本覆盖率 ---")
fc_df = df_orig[df_orig['final_candidate']==True]
if 'cdr3_sequence' in fc_df.columns:
    fc_sequences = fc_df['cdr3_sequence'].dropna().tolist()
    
    # v2.4策略下FC正样本是否通过硬约束
    v24_covered = 0
    v30_covered = 0
    for seq in fc_sequences:
        # v2.4: 首残基必须是F/W/Y
        v24_pass = seq[0] in v24_strategy['hard_constraints'].get('cdr3_first_residue_whitelist', ['F','W','Y'])
        # v3.0: 首残基白名单扩展
        v30_pass = seq[0] in v30_strategy['hard_constraints'].get('cdr3_first_residue_whitelist', ['F','W','Y','V','A','D','T'])
        
        if v24_pass:
            v24_covered += 1
        if v30_pass:
            v30_covered += 1
    
    print(f"  FC正样本 (n={len(fc_sequences)}):")
    print(f"    v2.4首残基白名单覆盖率: {v24_covered}/{len(fc_sequences)} ({v24_covered/len(fc_sequences):.1%})")
    print(f"    v3.0首残基白名单覆盖率: {v30_covered}/{len(fc_sequences)} ({v30_covered/len(fc_sequences):.1%})")
    print(f"    覆盖率提升: +{(v30_covered-v24_covered)/len(fc_sequences):.1%}")

# 6. Shannon熵
print(f"\n--- 6. 序列多样性 ---")
def shannon_entropy(sequences):
    """计算序列集合的Shannon熵"""
    from collections import Counter
    counts = Counter(sequences)
    total = len(sequences)
    probs = [c/total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

v24_entropy = shannon_entropy(v24_df['cdr3'].tolist())
v30_entropy = shannon_entropy(v30_df['cdr3'].tolist())
print(f"  v2.4 Shannon熵: {v24_entropy:.4f}")
print(f"  v3.0 Shannon熵: {v30_entropy:.4f}")
print(f"  ΔShannon: {v30_entropy-v24_entropy:+.4f}")

# === 保存结果 ===
v24_df.to_csv(f'{output_dir}/v24_generated_scored.csv', index=False)
v30_df.to_csv(f'{output_dir}/v30_generated_scored.csv', index=False)

# === 汇总 ===
print(f"\n{'='*70}")
print(f"=== 汇总 ===")
print(f"{'='*70}")
print(f"""
v2.4 (当前):
  序列数: {len(v24_df)}
  首残基芳香族率: {v24_aromatic_rate:.1%}
  RF2评分均值: {v24_df['rf2_score'].mean():.3f}
  Final评分均值: {v24_df['final_score'].mean():.3f}
  组合评分均值: {v24_df['combined_score'].mean():.3f}
  Shannon熵: {v24_entropy:.4f}

v3.0 (漏斗感知):
  序列数: {len(v30_df)}
  首残基芳香族率: {v30_aromatic_rate:.1%}
  RF2评分均值: {v30_df['rf2_score'].mean():.3f}
  Final评分均值: {v30_df['final_score'].mean():.3f}
  组合评分均值: {v30_df['combined_score'].mean():.3f}
  Shannon熵: {v30_entropy:.4f}

关键差异:
  RF2评分: v3.0 {'低于' if v30_df['rf2_score'].mean() < v24_df['rf2_score'].mean() else '高于'}v2.4 (预期: 非芳香族首残基降低RF2通过率)
  Final评分: v3.0 {'高于' if v30_df['final_score'].mean() > v24_df['final_score'].mean() else '低于'}v2.4 (预期: 非芳香族首残基提升Final概率)
  组合评分: v3.0 {'高于' if v30_df['combined_score'].mean() > v24_df['combined_score'].mean() else '低于'}v2.4
  Shannon熵: v3.0 {'高于' if v30_entropy > v24_entropy else '低于'}v2.4 (预期: 扩展白名单增加多样性)
""")
