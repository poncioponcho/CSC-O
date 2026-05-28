import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = './analyze1/output'
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

fig = plt.figure(figsize=(24, 20))
fig.suptitle('CSC-O: Causal-Stratified Counterfactual Optimization Dashboard\n10572 Antibody Sequences | Q02223 Target',
             fontsize=18, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(3, 3, 1)
stages = ['RF2\nFailed', 'AF3\nFailed', 'Schrod.\nFailed', 'Desmond\nFailed', 'Final\nCandidate']
counts = [9337, 1173, 0, 0, 62]
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
ax1.bar(stages, counts, color=colors)
ax1.set_title('Pipeline Attrition', fontweight='bold')
ax1.set_ylabel('Sequences')
for i, v in enumerate(counts):
    ax1.text(i, v + 100, str(v), ha='center', fontsize=8)

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

ax3 = fig.add_subplot(3, 3, 3)
variables = ['Glycine\nRatio', 'Serine\nRatio', 'CDR3\nLength', 'Hydrophobic\nRatio', 'Positive\nRatio', 'Aromatic\nRatio']
hrs = [4.35, 1.77, 1.08, 1.18, 0.95, 0.67]
ci_l = [3.46, 1.46, 1.07, 0.99, 0.73, 0.52]
ci_u = [5.47, 2.14, 1.09, 1.42, 1.24, 0.85]
y_pos = range(len(variables))
ax3.errorbar(hrs, y_pos, xerr=[np.array(hrs) - np.array(ci_l), np.array(ci_u) - np.array(hrs)],
             fmt='o', color='#2c3e50', ecolor='#7f8c8d', capsize=3)
ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(variables, fontsize=8)
ax3.set_xlabel('Hazard Ratio')
ax3.set_title('Cox PH: Hazard Ratios', fontweight='bold')
ax3.set_xscale('log')

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

ax5 = fig.add_subplot(3, 3, 5)
ate_vars = ['First Aromatic\n(Y/W/F)', 'Aromatic\nRatio', 'Glycine\nRatio', 'Proline\nCount', 'Serine\nRatio']
ate_vals = [-6.54, -1.02, -0.94, 1.35, 9.69]
colors_ate = ['#2ecc71' if v < 0 else '#e74c3c' for v in ate_vals]
ax5.barh(ate_vars, ate_vals, color=colors_ate, alpha=0.8)
ax5.axvline(x=0, color='black', linewidth=0.5)
ax5.set_xlabel('ATE on Interaction PAE')
ax5.set_title('Causal ATE Estimates', fontweight='bold')

ax6 = fig.add_subplot(3, 3, 6)
pos_cate_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'position_specific_cate.csv'))
if len(pos_cate_df) > 0:
    pivot = pos_cate_df.pivot_table(index='amino_acid', columns='position', values='CATE', aggfunc='first')
    sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax6, cbar_kws={'label': 'CATE', 'shrink': 0.8})
ax6.set_title('Position-Specific CATE', fontweight='bold')
ax6.set_xlabel('CDR3 Position')
ax6.set_ylabel('Amino Acid')

ax7 = fig.add_subplot(3, 3, 7)
cf_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'counterfactual_suggestions.csv'))
if len(cf_df) > 0:
    top_edits = cf_df['edit'].value_counts().head(12)
    ax7.barh(range(len(top_edits)), top_edits.values, color='#2ecc71', alpha=0.8)
    ax7.set_yticks(range(len(top_edits)))
    ax7.set_yticklabels(top_edits.index, fontsize=8)
ax7.set_xlabel('Frequency')
ax7.set_title('Top Salvage Edits', fontweight='bold')

ax8 = fig.add_subplot(3, 3, 8)
anti_data = {
    'Pattern': ['SSS', 'GGG', 'LL', 'CDR3>=10'],
    'With Pattern': [1.5, 5.6, 5.6, 2.6],
    'Without Pattern': [12.2, 11.8, 11.7, 30.8],
}
anti_df = pd.DataFrame(anti_data)
x = np.arange(len(anti_df))
ax8.bar(x - 0.2, anti_df['With Pattern'], 0.4, label='With Pattern', color='#e74c3c')
ax8.bar(x + 0.2, anti_df['Without Pattern'], 0.4, label='Without Pattern', color='#2ecc71')
ax8.set_xticks(x)
ax8.set_xticklabels(anti_df['Pattern'])
ax8.set_ylabel('RF2 Pass Rate (%)')
ax8.set_title('Anti-Pattern Impact', fontweight='bold')
ax8.legend(fontsize=8)

ax9 = fig.add_subplot(3, 3, 9)
metrics = ['Current\nRF2 Pass', 'Predicted\nRF2 Pass', 'Current\nCandidate', 'Predicted\nCandidate']
values = [11.7, 42.5, 0.59, 5.5]
colors_pred = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']
ax9.bar(metrics, values, color=colors_pred, alpha=0.8)
ax9.set_ylabel('Rate (%)')
ax9.set_title('Predicted Improvement', fontweight='bold')
for i, v in enumerate(values):
    ax9.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
dashboard_path = os.path.join(OUTPUT_DIR, 'csco_dashboard.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Dashboard saved to {dashboard_path}')

print('\n所有可视化完成！')
