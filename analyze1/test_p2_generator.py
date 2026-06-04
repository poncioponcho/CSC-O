#!/usr/bin/env python3
"""P2单元测试: csco_funnel_generator.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csco_funnel_generator import FunnelAwareGenerator, run_funnel_generation
from csco_funnel_aware_strategy import FunnelAwareStrategy
import pandas as pd
import numpy as np

print("="*60)
print("P2 单元测试: csco_funnel_generator.py")
print("="*60)

strategy = FunnelAwareStrategy(verbose=False)
gen = FunnelAwareGenerator(strategy=strategy, verbose=False)

# 1. 初始化验证
print("\n--- 1. 初始化验证 ---")
try:
    FunnelAwareGenerator(min_edit_distance=-1)
    print("  ❌ 负编辑距离应抛ValueError")
except ValueError:
    print("  ✅ 无效min_edit_distance正确抛出")

try:
    FunnelAwareGenerator(min_edit_distance=20)
    print("  ❌ 超范围编辑距离应抛ValueError")
except ValueError:
    print("  ✅ 超范围min_edit_distance正确抛出")

# 2. 参数验证
print("\n--- 2. 参数验证 ---")
for param, val, label in [('n_samples', 0, '零'), ('top_n', -1, '负'), ('min_combined_score', -0.1, '负分数')]:
    try:
        if param == 'n_samples':
            gen.generate(n_samples=val)
        elif param == 'top_n':
            gen.generate(top_n=val)
        elif param == 'min_combined_score':
            gen.generate(min_combined_score=val)
        print(f"  ❌ {label}{param}应抛ValueError")
    except ValueError:
        print(f"  ✅ {label}{param}正确拦截")

# 3. 核心生成功能
print("\n--- 3. 核心生成功能 ---")
seqs = gen.generate(n_samples=500, top_n=50, seed=42)
print(f"  生成序列数: {len(seqs)}")
print(f"  ✅ 生成成功" if len(seqs) > 0 else "  ❌ 生成失败")

if seqs:
    # 检查输出格式
    d = seqs[0].to_dict()
    required = ['sequence', 'length', 'first_aa', 'predicted_p_rf2',
                'predicted_p_final_given_rf2', 'predicted_p_final',
                'combined_score', 'diversity_score']
    missing = [k for k in required if k not in d]
    print(f"  ✅ 输出格式完整" if not missing else f"  ❌ 缺少: {missing}")

# 4. 概率校准
print("\n--- 4. 概率校准 ---")
if seqs:
    p_rf2_mean = np.mean([s.estimated_p_rf2 for s in seqs])
    p_final_mean = np.mean([s.estimated_p_final for s in seqs])
    p_final_g_rf2_mean = np.mean([s.estimated_p_final_given_rf2 for s in seqs])
    print(f"  P(RF2)均值: {p_rf2_mean:.4f}")
    print(f"  P(final|RF2)均值: {p_final_g_rf2_mean:.4f}")
    print(f"  P(final)均值: {p_final_mean:.6f}")
    
    # P(final) = P(RF2) × P(final|RF2)
    product = p_rf2_mean * p_final_g_rf2_mean
    print(f"  P(RF2)×P(final|RF2)={product:.6f} vs P(final)={p_final_mean:.6f}")
    print(f"  ✅ 分解建模一致性" if abs(product - p_final_mean) < 0.001 else "  ⚠️ 分解建模偏差")

# 5. 多样性过滤
print("\n--- 5. 多样性过滤 ---")
gen_no_div = FunnelAwareGenerator(strategy=strategy, min_edit_distance=0, verbose=False)
gen_div3 = FunnelAwareGenerator(strategy=strategy, min_edit_distance=3, verbose=False)
seqs_no_div = gen_no_div.generate(n_samples=500, top_n=50, seed=42)
seqs_div3 = gen_div3.generate(n_samples=500, top_n=50, seed=42)
print(f"  无多样性过滤: {len(seqs_no_div)} 条")
print(f"  编辑距离≥3: {len(seqs_div3)} 条")

if seqs_div3:
    # 检查多样性
    div_scores = [s.diversity_score for s in seqs_div3]
    print(f"  平均多样性评分: {np.mean(div_scores):.2f}")

# 6. 编辑距离
print("\n--- 6. 编辑距离 ---")
ed = FunnelAwareGenerator._edit_distance("WADKEY", "WADKEY")
print(f"  相同序列: ed={ed} (应为0)")
ed2 = FunnelAwareGenerator._edit_distance("WADKEY", "FADKEY")
print(f"  单点突变: ed={ed2} (应为1)")
ed3 = FunnelAwareGenerator._edit_distance("ABC", "XYZ")
print(f"  完全不同: ed={ed3} (应为3)")
print(f"  ✅ 编辑距离计算正确" if ed == 0 and ed2 == 1 and ed3 == 3 else "  ❌ 编辑距离计算错误")

# 7. 首残基分布
print("\n--- 7. 首残基分布 ---")
if seqs:
    first_aas = [s.first_aa for s in seqs]
    aa_counts = pd.Series(first_aas).value_counts()
    aromatic_rate = sum(1 for a in first_aas if a in 'FWY') / len(first_aas)
    print(f"  首残基分布: {dict(aa_counts.head(5))}")
    print(f"  芳香族率: {aromatic_rate:.1%}")
    print(f"  ✅ 首残基多样性" if aromatic_rate < 1.0 else "  ⚠️ 首残基全部芳香族")

# 8. CSV输出
print("\n--- 8. CSV输出 ---")
os.makedirs('output_v3_funnel', exist_ok=True)
df = gen.generate_to_csv(
    'output_v3_funnel/test_funnel_generated.csv',
    n_samples=500, top_n=50, seed=42,
)
print(f"  ✅ CSV输出: {len(df)} 条, 列={list(df.columns[:5])}..." if not df.empty else "  ❌ CSV输出失败")

# 9. 便捷函数
print("\n--- 9. 便捷函数 ---")
df2 = run_funnel_generation(
    strategy_path='output_v3_funnel/design_strategy_v3.0.json',
    output_dir='output_v3_funnel',
    n_samples=500, top_n=50, verbose=False,
)
print(f"  ✅ 便捷函数: {len(df2)} 条" if not df2.empty else "  ⚠️ 便捷函数无输出")

# 10. 大批量性能
print("\n--- 10. 大批量性能 ---")
import time
t0 = time.time()
seqs_large = gen.generate(n_samples=5000, top_n=200, seed=42)
elapsed = time.time() - t0
print(f"  5K采样→200序列: {elapsed:.1f}s, {len(seqs_large)} 条")
print(f"  ✅ 性能可接受" if elapsed < 60 else "  ⚠️ 性能需优化")

print("\n" + "="*60)
print("P2 单元测试完成")
print("="*60)
