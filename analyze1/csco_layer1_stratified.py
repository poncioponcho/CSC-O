import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

surv_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'survival_data.csv'))
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))

print('=' * 70)
print('第一层：分层归因与动态阈值校准')
print('=' * 70)

print('\n--- 1.1 管线 Attrition 模式分析 ---')
total = len(surv_df)
stage1_fail = (surv_df['time'] == 1) & (surv_df['event'] == 1)
stage2_fail = (surv_df['time'] == 2) & (surv_df['event'] == 1)
stage3_fail = (surv_df['time'] == 3) & (surv_df['event'] == 1)
stage4_fail = (surv_df['time'] == 4) & (surv_df['event'] == 1)
survived = (surv_df['event'] == 0)

print(f'总序列数: {total}')
print(f'RF2阶段失败: {stage1_fail.sum()} ({stage1_fail.sum()/total*100:.1f}%)')
print(f'AF3阶段失败: {stage2_fail.sum()} ({stage2_fail.sum()/total*100:.1f}%)')
print(f'Schrödinger阶段失败: {stage3_fail.sum()} ({stage3_fail.sum()/total*100:.1f}%)')
print(f'Desmond阶段失败: {stage4_fail.sum()} ({stage4_fail.sum()/total*100:.1f}%)')
print(f'最终候选: {survived.sum()} ({survived.sum()/total*100:.1f}%)')

rf2_pass_count = (~stage1_fail).sum()
af3_fail_of_rf2pass = stage2_fail.sum()
schrod_fail_of_af3pass = stage3_fail.sum()
desmond_fail_of_schrodpass = stage4_fail.sum()

print(f'\n--- 条件失败率 ---')
print(f'RF2通过后AF3失败: {af3_fail_of_rf2pass}/{rf2_pass_count} ({af3_fail_of_rf2pass/rf2_pass_count*100:.1f}%)')
if rf2_pass_count - af3_fail_of_rf2pass > 0:
    af3_pass_count = rf2_pass_count - af3_fail_of_rf2pass
    print(f'AF3通过后Schrödinger失败: {schrod_fail_of_af3pass}/{af3_pass_count} ({schrod_fail_of_af3pass/af3_pass_count*100:.1f}%)')
    if af3_pass_count - schrod_fail_of_af3pass > 0:
        schrod_pass_count = af3_pass_count - schrod_fail_of_af3pass
        print(f'Schrödinger通过后Desmond失败: {desmond_fail_of_schrodpass}/{schrod_pass_count} ({desmond_fail_of_schrodpass/schrod_pass_count*100:.1f}%)')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

stages = ['RF2', 'AF3', 'Schrödinger', 'Desmond', 'Candidate']
counts = [stage1_fail.sum(), stage2_fail.sum(), stage3_fail.sum(), stage4_fail.sum(), survived.sum()]
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']

axes[0].bar(stages, counts, color=colors)
axes[0].set_title('Attrition by Pipeline Stage', fontsize=14)
axes[0].set_ylabel('Number of Sequences')
for i, v in enumerate(counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=10)

cumulative_stages = ['Start'] + stages
cumulative = [total]
remaining = total
for c in counts:
    remaining -= c
    cumulative.append(max(remaining, 0))

axes[1].plot(cumulative_stages, cumulative, 'o-', color='#2c3e50', linewidth=2, markersize=8)
axes[1].fill_between(range(len(cumulative_stages)), cumulative, alpha=0.15, color='#3498db')
axes[1].set_title('Cumulative Survival Through Pipeline', fontsize=14)
axes[1].set_ylabel('Remaining Sequences')
axes[1].set_ylim(0, total * 1.05)
for i, v in enumerate(cumulative):
    axes[1].text(i, v + 200, str(v), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_attrition_funnel.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nAttrition funnel plot saved.')

print('\n--- 1.2 Cox比例风险模型 ---')

cox_df = surv_df.copy()
cox_df['cdr3_len_centered'] = cox_df['cdr3_len'] - cox_df['cdr3_len'].mean()
cox_df['positive_ratio_centered'] = cox_df['positive_ratio'] - cox_df['positive_ratio'].mean()
cox_df['aromatic_ratio_centered'] = cox_df['aromatic_ratio'] - cox_df['aromatic_ratio'].mean()
cox_df['glycine_ratio_centered'] = cox_df['glycine_ratio'] - cox_df['glycine_ratio'].mean()
cox_df['serine_ratio_centered'] = cox_df['serine_ratio'] - cox_df['serine_ratio'].mean()
cox_df['hydrophobic_ratio_centered'] = cox_df['hydrophobic_ratio'] - cox_df['hydrophobic_ratio'].mean()

cox_vars = ['cdr3_len_centered', 'positive_ratio_centered', 'aromatic_ratio_centered',
            'glycine_ratio_centered', 'serine_ratio_centered', 'hydrophobic_ratio_centered']
cox_data = cox_df[['time', 'event'] + cox_vars].dropna()

cph = CoxPHFitter()
cph.fit(cox_data, duration_col='time', event_col='event')

print('\nCox PH Model Summary:')
cph.print_summary()

hr_results = []
for var in cox_vars:
    coef = cph.params_[var]
    hr = np.exp(coef)
    ci_lower = np.exp(cph.confidence_intervals_.loc[var, '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc[var, '95% upper-bound'])
    p_val = cph.summary.loc[var, 'p']
    hr_results.append({
        'variable': var.replace('_centered', ''),
        'coef': coef,
        'HR': hr,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_val,
    })

hr_df = pd.DataFrame(hr_results).sort_values('HR', ascending=False)
hr_df.to_csv(os.path.join(OUTPUT_DIR, 'cox_hazard_ratios.csv'), index=False)

print('\n--- Hazard Ratios (sorted) ---')
for _, row in hr_df.iterrows():
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    print(f"  {row['variable']:25s} HR={row['HR']:.3f} [{row['CI_lower']:.3f}-{row['CI_upper']:.3f}] p={row['p_value']:.4f} {sig}")

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(hr_df))
xerr_lower = hr_df['HR'] - hr_df['CI_lower']
xerr_upper = hr_df['CI_upper'] - hr_df['HR']
ax.errorbar(hr_df['HR'], y_pos, xerr=[xerr_lower, xerr_upper], fmt='o', color='#2c3e50', ecolor='#7f8c8d', capsize=4)
ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(hr_df['variable'])
ax.set_xlabel('Hazard Ratio (95% CI)')
ax.set_title('Cox PH Model: Feature Hazard Ratios for Pipeline Survival', fontsize=13)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_forest_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Forest plot saved.')

print('\n--- 1.3 Kaplan-Meier生存曲线（按CDR3长度分组）---')

fig, ax = plt.subplots(figsize=(10, 7))
kmf = KaplanMeierFitter()

len_groups = {
    '5-7 (short)': surv_df['cdr3_len'].isin([5, 6, 7]),
    '8-9 (medium)': surv_df['cdr3_len'].isin([8, 9]),
    '10-13 (long)': surv_df['cdr3_len'].isin([10, 11, 12, 13]),
}

for label, mask in len_groups.items():
    group_data = surv_df[mask]
    kmf.fit(group_data['time'], group_data['event'], label=label)
    kmf.plot_survival_function(ax=ax)

ax.set_xlabel('Pipeline Stage')
ax.set_ylabel('Survival Probability')
ax.set_title('Kaplan-Meier Survival by CDR3 Length Group', fontsize=13)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['RF2', 'AF3', 'Schrödinger', 'Desmond'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_km_by_cdr3_length.png'), dpi=150, bbox_inches='tight')
plt.close()
print('KM plot saved.')

print('\n--- 1.4 RF2阈值敏感性分析 ---')

lddt_values = feat_df['rf2_pred_lddt'].values.astype(float)
pae_values = feat_df['rf2_interaction_pae'].values.astype(float)
rf2_passed = feat_df['rf2_passed'].values.astype(bool)
final_candidates = feat_df['final_candidate'].values.astype(bool)

thresholds = np.arange(0.82, 0.92, 0.005)
sensitivity_results = []

for thresh in thresholds:
    pred_pass = (lddt_values >= thresh) & (pae_values <= 10.0)
    tp = np.sum(pred_pass & final_candidates)
    fp = np.sum(pred_pass & ~final_candidates)
    fn = np.sum(~pred_pass & final_candidates)
    tn = np.sum(~pred_pass & ~final_candidates)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    pass_rate = pred_pass.sum() / len(pred_pass)

    sensitivity_results.append({
        'threshold': thresh,
        'pass_count': pred_pass.sum(),
        'pass_rate': pass_rate,
        'sensitivity': sensitivity,
        'precision': precision,
        'fpr': fpr,
    })

sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv(os.path.join(OUTPUT_DIR, 'threshold_sensitivity.csv'), index=False)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(sens_df['threshold'], sens_df['pass_rate'], 'b-o', markersize=4, label='Pass Rate')
ax1.set_xlabel('RF2 pred_lddt Threshold')
ax1.set_ylabel('Pass Rate', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(sens_df['threshold'], sens_df['precision'], 'r-s', markersize=4, label='Precision (final candidate)')
ax2.set_ylabel('Precision', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax1.axvline(x=0.88, color='green', linestyle='--', alpha=0.7, label='Current threshold (0.88)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_title('RF2 Threshold Sensitivity: Pass Rate vs Precision', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_threshold_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Threshold sensitivity plot saved.')

print('\n--- 1.5 多目标优化权重建议 ---')

rf2_pass_df = feat_df[feat_df['rf2_passed'] == True].copy()
print(f'\nRF2通过样本数: {len(rf2_pass_df)}')
print(f'其中最终候选数: {rf2_pass_df["final_candidate"].sum()}')
print(f'RF2通过→最终候选转化率: {rf2_pass_df["final_candidate"].sum()/len(rf2_pass_df)*100:.1f}%')

print('\n--- 1.6 RF2失败原因精细分析 ---')
reason_counts = feat_df['rf2_filter_reason'].value_counts(dropna=False)
print(reason_counts)

print('\n--- 1.7 CDR3长度与RF2通过率关系 ---')
len_pass = feat_df.groupby('cdr3_len').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
    final_cand=('final_candidate', 'sum'),
).reset_index()
len_pass['rf2_pass_rate'] = len_pass['rf2_pass'] / len_pass['total'] * 100
len_pass['candidate_rate'] = len_pass['final_cand'] / len_pass['total'] * 100
print(len_pass.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(len_pass['cdr3_len'] - 0.2, len_pass['rf2_pass_rate'], 0.4, label='RF2 Pass Rate (%)', color='#3498db')
ax1.bar(len_pass['cdr3_len'] + 0.2, len_pass['candidate_rate'], 0.4, label='Final Candidate Rate (%)', color='#2ecc71')
ax1.set_xlabel('CDR3 Length')
ax1.set_ylabel('Rate (%)')
ax1.set_title('RF2 Pass Rate and Final Candidate Rate by CDR3 Length', fontsize=13)
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(len_pass['cdr3_len'], len_pass['total'], 'r--o', label='Total Count')
ax2.set_ylabel('Total Count', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_cdr3_length_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print('CDR3 length analysis plot saved.')

print('\n--- 1.8 首尾残基与RF2通过率 ---')
first_res_pass = feat_df.groupby('first_residue').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
).reset_index()
first_res_pass['rate'] = first_res_pass['rf2_pass'] / first_res_pass['total'] * 100
first_res_pass = first_res_pass.sort_values('rate', ascending=False)
print('\nCDR3首残基通过率:')
print(first_res_pass.to_string(index=False))

last_res_pass = feat_df.groupby('last_residue').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
).reset_index()
last_res_pass['rate'] = last_res_pass['rf2_pass'] / last_res_pass['total'] * 100
last_res_pass = last_res_pass.sort_values('rate', ascending=False)
print('\nCDR3尾残基通过率:')
print(last_res_pass.to_string(index=False))

print('\n' + '=' * 70)
print('第一层分析完成！所有结果已保存至 output/ 目录')
print('=' * 70)
