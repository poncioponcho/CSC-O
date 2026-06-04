#!/usr/bin/env python3
"""P1单元测试: csco_funnel_counterfactual.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csco_funnel_counterfactual import FunnelAwareCounterfactual, run_counterfactual_analysis
from csco_funnel_aware_strategy import FunnelAwareStrategy
from csco_config import AROMATIC
import pandas as pd

print("="*60)
print("P1 单元测试: csco_funnel_counterfactual.py")
print("="*60)

strategy = FunnelAwareStrategy(verbose=False)
nav = FunnelAwareCounterfactual(strategy=strategy, verbose=False)

# 1. 初始化
print("\n--- 1. 初始化 ---")
try:
    FunnelAwareCounterfactual(method='invalid')
    print("  ❌ 应抛ValueError")
except ValueError:
    print("  ✅ 无效method正确抛出")

# 2. 输入验证
print("\n--- 2. 输入验证 ---")
for seq, label in [("", "空"), ("ABCD", "短"), (12345, "非字符串")]:
    try:
        nav.suggest_mutation(seq)
        print(f"  ❌ {label}输入应抛ValueError")
    except (ValueError, TypeError):
        print(f"  ✅ {label}输入正确拦截")

try:
    nav.suggest_mutation("WADKEY", top_k=0)
    print("  ❌ top_k=0应抛ValueError")
except ValueError:
    print("  ✅ 无效top_k正确抛出")

# 3. 核心功能
print("\n--- 3. 核心功能 ---")
for seq in ["WADKEY", "GADKEY", "WADKEYA"]:
    r = nav.suggest_mutation(seq, top_k=5)
    best = f"{r.best_mutation.mutation} ΔP(final)={r.best_mutation.delta_p_final:+.4f}" if r.best_mutation else "无"
    print(f"  {seq}: {r.n_mutations_evaluated}评估, {r.n_beneficial}有益, 最优={best}")

# 4. 阶段反转
print("\n--- 4. 阶段反转 ---")
r = nav.suggest_mutation("WADKEY", top_k=20)
aromatic_to_non = [s for s in r.suggestions if s.position == 0 and s.original_aa in AROMATIC and s.new_aa not in AROMATIC]
if aromatic_to_non:
    s = aromatic_to_non[0]
    rf2_down = s.rf2_score_delta < 0
    final_up = s.final_score_delta > 0
    print(f"  {s.mutation}: RF2Δ={s.rf2_score_delta:+.3f}({'↓' if rf2_down else '↑'}), FinalΔ={s.final_score_delta:+.3f}({'↑' if final_up else '↓'})")
    if rf2_down and final_up:
        print("  ✅ 阶段反转验证通过: 芳香族→非芳香族, RF2↓Final↑")

# 5. 位置限制
print("\n--- 5. 位置限制 ---")
r = nav.suggest_mutation("WADKEY", top_k=5, positions=[0])
all_pos0 = all(s.position == 0 for s in r.suggestions)
print(f"  ✅ 位置限制正确" if all_pos0 else "  ❌ 位置限制失败")

# 6. 氨基酸限制
print("\n--- 6. 氨基酸限制 ---")
r = nav.suggest_mutation("WADKEY", top_k=5, allowed_aa=['F', 'G'])
all_allowed = all(s.new_aa in ['F', 'G'] for s in r.suggestions)
print(f"  ✅ 氨基酸限制正确" if all_allowed else "  ❌ 氨基酸限制失败")

# 7. 批量分析
print("\n--- 7. 批量分析 ---")
batch = nav.suggest_mutation_batch(["WADKEY", "GADKEY", "FKDSPY"], top_k=3)
print(f"  ✅ 批量分析: {len(batch)} 条成功")
try:
    nav.suggest_mutation_batch([])
    print("  ❌ 空列表应抛ValueError")
except ValueError:
    print("  ✅ 空列表正确拦截")

# 8. 输出格式
print("\n--- 8. 输出格式 ---")
r = nav.suggest_mutation("WADKEY", top_k=3)
if r.best_mutation:
    d = r.best_mutation.to_dict()
    required = ['mutation', 'position', 'original_aa', 'new_aa', 'mutated_sequence',
                'delta_p_rf2', 'delta_p_final_given_rf2', 'delta_p_final', 'confidence', 'evidence']
    missing = [k for k in required if k not in d]
    print(f"  ✅ 输出格式完整" if not missing else f"  ❌ 缺少: {missing}")

# 9. FC正样本
print("\n--- 9. FC正样本 ---")
df_orig = pd.read_csv('output_server_v2.3/feature_matrix.csv')
fc_df = df_orig[df_orig['final_candidate'] == True]
if 'cdr3_sequence' in fc_df.columns and len(fc_df) > 0:
    for seq in fc_df['cdr3_sequence'].dropna().tolist()[:3]:
        r = nav.suggest_mutation(seq, top_k=3)
        best = f"{r.best_mutation.mutation} ΔP(final)={r.best_mutation.delta_p_final:+.4f}" if r.best_mutation else "无"
        print(f"  {seq}: {r.n_beneficial}有益, 最优={best}")

# 10. 便捷函数
print("\n--- 10. 便捷函数 ---")
os.makedirs('output_v3_funnel', exist_ok=True)
rdf = run_counterfactual_analysis(["WADKEY", "GADKEY"], 'output_v3_funnel', top_k=3, verbose=False)
print(f"  ✅ 便捷函数: {len(rdf)} 条建议" if not rdf.empty else "  ⚠️ 无建议")

print("\n" + "="*60)
print("P1 单元测试完成")
print("="*60)
