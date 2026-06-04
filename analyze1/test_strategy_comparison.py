#!/usr/bin/env python3
"""Step C: v2.4 vs v3.0 策略对比模拟"""
import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csco_funnel_aware_strategy import FunnelAwareStrategy, DEFAULT_STAGE_AWARE_CONSTRAINTS

# 加载原始数据
df = pd.read_csv('output_server_v2.3/feature_matrix.csv')
print(f"数据: {len(df)} 条序列")

# 初始化漏斗感知策略
strategy = FunnelAwareStrategy(verbose=True)

# === 1. 对全部10,572条序列进行v2.4 vs v3.0对比 ===
print("\n=== 全部序列 v2.4 vs v3.0 对比 ===")

# 获取CDR3序列
if 'cdr3_sequence' in df.columns:
    sequences = df['cdr3_sequence'].dropna().tolist()
else:
    print("错误: 找不到cdr3_sequence列")
    sys.exit(1)

comparison_df = strategy.compare_strategies(sequences)

# 统计
n_total = len(comparison_df)
n_divergence = comparison_df['strategy_divergence'].sum()
n_aromatic = comparison_df['first_is_aromatic'].sum()
n_non_aromatic_v3 = ((~comparison_df['first_is_aromatic']) & (comparison_df['v3_prefers'])).sum()

print(f"\n策略分歧: {n_divergence}/{n_total} ({n_divergence/n_total*100:.1f}%)")
print(f"芳香族首残基: {n_aromatic} ({n_aromatic/n_total*100:.1f}%)")
print(f"非芳香族但v3.0偏好: {n_non_aromatic_v3}")

# === 2. FC正样本分析 ===
print("\n=== FC正样本 (n=65) 分析 ===")
fc_mask = df['final_candidate'] == True
fc_df = df[fc_mask]

if 'cdr3_sequence' in fc_df.columns:
    fc_sequences = fc_df['cdr3_sequence'].dropna().tolist()
    fc_comparison = strategy.compare_strategies(fc_sequences)
    
    fc_aromatic_rate = fc_comparison['first_is_aromatic'].mean()
    fc_v3_preferred = fc_comparison['v3_prefers'].mean()
    fc_v3_combined_mean = fc_comparison['v3_combined_score'].mean()
    
    print(f"FC正样本首残基芳香族率: {fc_aromatic_rate:.1%}")
    print(f"FC正样本v3.0偏好率: {fc_v3_preferred:.1%}")
    print(f"FC正样本v3.0组合得分均值: {fc_v3_combined_mean:.3f}")
    
    # FC正样本首残基分布
    fc_first_aa = fc_comparison['first_aa'].value_counts()
    print(f"\nFC正样本首残基分布:")
    for aa, count in fc_first_aa.items():
        is_aromatic = "芳香族" if aa in 'FWY' else "非芳香族"
        print(f"  {aa}: {count} ({is_aromatic})")

# === 3. RF2通过但非FC的样本分析 ===
print("\n=== RF2通过但非FC样本 (n=1,170右删失) 分析 ===")
rf2_pass_no_fc = df[(df['rf2_passed'] == True) & (df['final_candidate'] == False)]
if 'cdr3_sequence' in rf2_pass_no_fc.columns:
    rf2_sequences = rf2_pass_no_fc['cdr3_sequence'].dropna().tolist()
    rf2_comparison = strategy.compare_strategies(rf2_sequences)
    
    rf2_aromatic_rate = rf2_comparison['first_is_aromatic'].mean()
    rf2_v3_preferred = rf2_comparison['v3_prefers'].mean()
    
    print(f"RF2通过非FC首残基芳香族率: {rf2_aromatic_rate:.1%}")
    print(f"RF2通过非FC v3.0偏好率: {rf2_v3_preferred:.1%}")

# === 4. 生成v3.0策略文件 ===
print("\n=== 生成v3.0策略 ===")
os.makedirs('output_v3_funnel', exist_ok=True)

v3_strategy = strategy.save_v3_strategy(
    'output_v3_funnel/design_strategy_v3.0.json',
    base_strategy_path='output_v2.4_test/design_strategy_v2.4.json',
)

# === 5. 保存对比结果 ===
comparison_df.to_csv('output_v3_funnel/strategy_comparison_v2_vs_v3.csv', index=False)
if 'cdr3_sequence' in fc_df.columns:
    fc_comparison.to_csv('output_v3_funnel/fc_positive_strategy_comparison.csv', index=False)

# === 6. 关键指标汇总 ===
print("\n" + "="*60)
print("=== v2.4 vs v3.0 关键差异汇总 ===")
print("="*60)

print(f"""
v2.4策略 (当前):
  - 首残基: 硬约束 [F, W, Y] (仅芳香族)
  - 优化目标: RF2通过率
  - 问题: first_is_aromatic Final HR=3.676 [2.42-5.58] (p=1e-9)

v3.0策略 (漏斗感知):
  - 首残基: 软偏好 [F, W, Y, V, A, D, T] (扩展白名单)
  - 优化目标: P(final) = P(RF2) × P(final|RF2)
  - 阶段权重: RF2=0.4, Final=0.6
  - 关键变更: cdr3_min_aromatic_first: True → False

数据支撑:
  - FC正样本芳香族率: {fc_aromatic_rate:.1%}
  - FC正样本v3.0偏好率: {fc_v3_preferred:.1%}
  - 策略分歧序列: {n_divergence/n_total*100:.1f}%
""")
