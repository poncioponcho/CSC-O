import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))
surv_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'survival_data.csv'))

# 动态计算核心指标
n_total = len(feat_df)
n_rf2_failed = int(((surv_df['time'] == 1) & (surv_df['event'] == 1)).sum())
n_af3_failed = int(((surv_df['time'] == 2) & (surv_df['event'] == 1)).sum())
n_schrod_failed = int(((surv_df['time'] == 3) & (surv_df['event'] == 1)).sum())
n_desmond_failed = int(((surv_df['time'] == 4) & (surv_df['event'] == 1)).sum())
n_candidate = int((surv_df['event'] == 0).sum())

current_rf2_pass_rate = feat_df['rf2_passed'].mean() * 100
current_candidate_rate = feat_df['final_candidate'].mean() * 100

# 动态标题
target_info = ''
if 'target_id' in feat_df.columns and feat_df['target_id'].notna().any():
    target_info = f" | {feat_df['target_id'].iloc[0]} Target"
fig_title = f'CSC-O: Causal-Stratified Counterfactual Optimization Dashboard\n{n_total} Antibody Sequences{target_info}'

fig = plt.figure(figsize=(24, 20))
fig.suptitle(fig_title, fontsize=18, fontweight='bold', y=0.98)

# ═══════════════════════════════════════════════════════════════
# 1) Pipeline Attrition
# ═══════════════════════════════════════════════════════════════

ax1 = fig.add_subplot(3, 3, 1)
stages = ['RF2\nFailed', 'AF3\nFailed', 'Schrod.\nFailed', 'Desmond\nFailed', 'Final\nCandidate']
counts = [n_rf2_failed, n_af3_failed, n_schrod_failed, n_desmond_failed, n_candidate]
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
ax1.bar(stages, counts, color=colors)
ax1.set_title('Pipeline Attrition', fontweight='bold')
ax1.set_ylabel('Sequences')
for i, v in enumerate(counts):
    ax1.text(i, v + max(counts) * 0.02, str(v), ha='center', fontsize=8)

# ═══════════════════════════════════════════════════════════════
# 2) CDR3 Length vs Pass Rate
# ═══════════════════════════════════════════════════════════════

ax2 = fig.add_subplot(3, 3, 2)
len_pass = feat_df.groupby('cdr3_len').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
    final_cand=('final_candidate', 'sum'),
).reset_index()
len_pass['pass_rate'] = len_pass['rf2_pass'] / len_pass['total'] * 100
len_pass['cand_rate'] = len_pass['final_cand'] / len_pass['total'] * 100
ax2.bar(len_pass['cdr3_len'] - 0.2, len_pass['pass_rate'], 0.4, label='RF2 Pass %', color='#3498db')
ax2.bar(len_pass['cdr3_len'] + 0.2, len_pass['cand_rate'], 0.4, label='Candidate %', color='#2ecc71')
ax2.set_xlabel('CDR3 Length')
ax2.set_ylabel('Rate (%)')
ax2.set_title('CDR3 Length vs Pass Rate', fontweight='bold')
ax2.legend(fontsize=8)

# ═══════════════════════════════════════════════════════════════
# 3) Cox PH Hazard Ratios
# ═══════════════════════════════════════════════════════════════

ax3 = fig.add_subplot(3, 3, 3)
cox_path = os.path.join(OUTPUT_DIR, 'cox_hazard_ratios.csv')
if os.path.exists(cox_path):
    cox_df = pd.read_csv(cox_path)
    cox_df = cox_df.sort_values('HR', ascending=True)
    y_pos = range(len(cox_df))
    hrs = cox_df['HR'].values
    ci_l = cox_df['CI_lower'].values
    ci_u = cox_df['CI_upper'].values
    labels = cox_df['variable'].str.replace('_', '\n').str.title().tolist()
    ax3.errorbar(hrs, y_pos,
                 xerr=[hrs - ci_l, ci_u - hrs],
                 fmt='o', color='#2c3e50', ecolor='#7f8c8d', capsize=3)
    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Hazard Ratio')
    ax3.set_title('Cox PH: Hazard Ratios', fontweight='bold')
    ax3.set_xscale('log')
else:
    ax3.text(0.5, 0.5, 'cox_hazard_ratios.csv not found', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Cox PH: Hazard Ratios', fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# 4) First Residue vs Pass Rate
# ═══════════════════════════════════════════════════════════════

ax4 = fig.add_subplot(3, 3, 4)
first_res = feat_df.groupby('first_residue').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
).reset_index()
first_res['rate'] = first_res['rf2_pass'] / first_res['total'] * 100
first_res = first_res.sort_values('rate', ascending=True)
first_res = first_res[first_res['total'] >= 40]
colors_fr = ['#e74c3c' if r < 10 else '#f1c40f' if r < 30 else '#2ecc71' for r in first_res['rate']]
ax4.barh(first_res['first_residue'], first_res['rate'], color=colors_fr)
ax4.set_xlabel('RF2 Pass Rate (%)')
ax4.set_title('First Residue vs Pass Rate', fontweight='bold')
ax4.axvline(x=50, color='red', linestyle='--', alpha=0.5)

# ═══════════════════════════════════════════════════════════════
# 5) Causal ATE Estimates
# ═══════════════════════════════════════════════════════════════

ax5 = fig.add_subplot(3, 3, 5)
ate_path = os.path.join(OUTPUT_DIR, 'ate_estimates.csv')
if os.path.exists(ate_path):
    ate_df = pd.read_csv(ate_path)
    # 只取对 interaction_pae 有显著效应的结果
    ate_sig = ate_df[(ate_df['outcome'] == 'rf2_interaction_pae') & (ate_df['p_value'] < 0.05)].copy()
    ate_sig = ate_sig.sort_values('ATE', ascending=True)
    if len(ate_sig) > 0:
        ate_labels = ate_sig['treatment'].str.replace('_', '\n').str.title().tolist()
        ate_vals = ate_sig['ATE'].values
        colors_ate = ['#2ecc71' if v < 0 else '#e74c3c' for v in ate_vals]
        ax5.barh(range(len(ate_vals)), ate_vals, color=colors_ate, alpha=0.8)
        ax5.axvline(x=0, color='black', linewidth=0.5)
        ax5.set_yticks(range(len(ate_vals)))
        ax5.set_yticklabels(ate_labels, fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No significant ATE found', ha='center', va='center', transform=ax5.transAxes)
else:
    ax5.text(0.5, 0.5, 'ate_estimates.csv not found', ha='center', va='center', transform=ax5.transAxes)
ax5.set_xlabel('ATE on Interaction PAE')
ax5.set_title('Causal ATE Estimates', fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# 6) Position-Specific CATE
# ═══════════════════════════════════════════════════════════════

ax6 = fig.add_subplot(3, 3, 6)
pos_cate_path = os.path.join(OUTPUT_DIR, 'position_specific_cate.csv')
if os.path.exists(pos_cate_path):
    pos_cate_df = pd.read_csv(pos_cate_path)
    if len(pos_cate_df) > 0:
        pivot = pos_cate_df.pivot_table(index='amino_acid', columns='position', values='CATE', aggfunc='first')
        sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax6, cbar_kws={'label': 'CATE', 'shrink': 0.8})
    else:
        ax6.text(0.5, 0.5, 'No position-specific CATE data', ha='center', va='center', transform=ax6.transAxes)
else:
    ax6.text(0.5, 0.5, 'position_specific_cate.csv not found', ha='center', va='center', transform=ax6.transAxes)
ax6.set_title('Position-Specific CATE', fontweight='bold')
ax6.set_xlabel('CDR3 Position')
ax6.set_ylabel('Amino Acid')

# ═══════════════════════════════════════════════════════════════
# 7) Top Salvage Edits
# ═══════════════════════════════════════════════════════════════

ax7 = fig.add_subplot(3, 3, 7)
cf_path = os.path.join(OUTPUT_DIR, 'counterfactual_suggestions.csv')
if os.path.exists(cf_path):
    cf_df = pd.read_csv(cf_path)
    if len(cf_df) > 0:
        top_edits = cf_df['edit'].value_counts().head(12)
        ax7.barh(range(len(top_edits)), top_edits.values, color='#2ecc71', alpha=0.8)
        ax7.set_yticks(range(len(top_edits)))
        ax7.set_yticklabels(top_edits.index, fontsize=8)
    else:
        ax7.text(0.5, 0.5, 'No counterfactual data', ha='center', va='center', transform=ax7.transAxes)
else:
    ax7.text(0.5, 0.5, 'counterfactual_suggestions.csv not found', ha='center', va='center', transform=ax7.transAxes)
ax7.set_xlabel('Frequency')
ax7.set_title('Top Salvage Edits', fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# 8) Anti-Pattern Impact
# ═══════════════════════════════════════════════════════════════

ax8 = fig.add_subplot(3, 3, 8)
patterns = ['SSS', 'GGG', 'LL', 'CDR3>=10']
with_rates = []
without_rates = []

for pattern, col in [('SSS', 'has_sss'), ('GGG', 'has_ggg'), ('LL', 'has_ll')]:
    if col in feat_df.columns and feat_df[col].sum() > 0:
        with_r = feat_df.loc[feat_df[col] == True, 'rf2_passed'].mean() * 100
        without_r = feat_df.loc[feat_df[col] == False, 'rf2_passed'].mean() * 100
    else:
        with_r, without_r = 0, 0
    with_rates.append(with_r)
    without_rates.append(without_r)

# CDR3>=10
with_r = feat_df.loc[feat_df['cdr3_len'] >= 10, 'rf2_passed'].mean() * 100
without_r = feat_df.loc[feat_df['cdr3_len'] < 10, 'rf2_passed'].mean() * 100
with_rates.append(with_r)
without_rates.append(without_r)

x = np.arange(len(patterns))
ax8.bar(x - 0.2, with_rates, 0.4, label='With Pattern', color='#e74c3c')
ax8.bar(x + 0.2, without_rates, 0.4, label='Without Pattern', color='#2ecc71')
ax8.set_xticks(x)
ax8.set_xticklabels(patterns)
ax8.set_ylabel('RF2 Pass Rate (%)')
ax8.set_title('Anti-Pattern Impact', fontweight='bold')
ax8.legend(fontsize=8)

# ═══════════════════════════════════════════════════════════════
# 9) Predicted Improvement
# ═══════════════════════════════════════════════════════════════

ax9 = fig.add_subplot(3, 3, 9)
# 当前值从数据动态计算；预测值来自设计策略的预期改进（保留为静态估计）
metrics = ['Current\nRF2 Pass', 'Predicted\nRF2 Pass', 'Current\nCandidate', 'Predicted\nCandidate']
# Predicted 值来自策略预期，若 design_strategy.json 存在则尝试读取
design_path = os.path.join(OUTPUT_DIR, 'design_strategy.json')
predicted_rf2_pass = 42.5
predicted_candidate = 5.5
if os.path.exists(design_path):
    import json
    with open(design_path) as f:
        strategy = json.load(f)
    # 若策略文件中有预测值则使用，否则保留默认值
    predicted_rf2_pass = strategy.get('predicted_rf2_pass_rate', predicted_rf2_pass)
    predicted_candidate = strategy.get('predicted_candidate_rate', predicted_candidate)

values = [current_rf2_pass_rate, predicted_rf2_pass, current_candidate_rate, predicted_candidate]
colors_pred = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']
ax9.bar(metrics, values, color=colors_pred, alpha=0.8)
ax9.set_ylabel('Rate (%)')
ax9.set_title('Predicted Improvement', fontweight='bold')
for i, v in enumerate(values):
    ax9.text(i, v + max(values) * 0.02, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
dashboard_path = os.path.join(OUTPUT_DIR, 'csco_dashboard.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Dashboard saved to {dashboard_path}')

print('\n所有可视化完成！')
