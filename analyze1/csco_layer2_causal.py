import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = './analyze1/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))

print('=' * 70)
print('第二层：因果约束引擎')
print('=' * 70)

print('\n--- 2.1 定义因果变量 ---')

treatment_vars = ['cdr3_len', 'positive_ratio', 'aromatic_ratio',
                  'glycine_ratio', 'serine_ratio', 'proline_count']
outcome_binary = 'rf2_passed'
outcome_continuous = 'rf2_interaction_pae'
confounder_vars = ['backbone_id']

feat_df['first_is_aromatic'] = feat_df['first_residue'].isin(['Y', 'W', 'F']).astype(int)
feat_df['last_is_YH'] = feat_df['last_residue'].isin(['Y', 'H']).astype(int)

treatment_vars_extended = treatment_vars + ['first_is_aromatic', 'last_is_YH']

print(f'处理变量: {treatment_vars_extended}')
print(f'结果变量(二元): {outcome_binary}')
print(f'结果变量(连续): {outcome_continuous}')
print(f'混杂变量: {confounder_vars}')

print('\n--- 2.2 PC算法因果图学习 ---')

causal_cols = treatment_vars_extended + [outcome_binary, outcome_continuous, 'backbone_id']
causal_data = feat_df[causal_cols].copy()
causal_data[outcome_binary] = causal_data[outcome_binary].astype(int)

def partial_corr_test(x, y, z_data, alpha=0.05):
    if z_data.shape[1] == 0:
        r, p = stats.pearsonr(x, y)
        return r, p

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(z_data, x)
    res_x = x - lr.predict(z_data)
    lr.fit(z_data, y)
    res_y = y - lr.predict(z_data)
    r, p = stats.pearsonr(res_x, res_y)
    return r, p

def pc_algorithm(data, alpha=0.01, domain_constraints=None):
    nodes = list(data.columns)
    n = len(nodes)
    adj = {i: set(range(n)) - {i} for i in range(n)}
    sep_sets = {}

    if domain_constraints is None:
        domain_constraints = []

    depth = 0
    max_depth = 3

    while depth <= max_depth:
        removed = []
        for i in range(n):
            for j in list(adj[i]):
                if j not in adj[i]:
                    continue
                neighbors = adj[i] - {j}
                if len(neighbors) < depth:
                    continue

                for cond_set in combinations(neighbors, depth):
                    cond_vars = [nodes[k] for k in cond_set]
                    z_data = data[cond_vars].values if cond_vars else np.empty((len(data), 0))

                    r, p = partial_corr_test(
                        data[nodes[i]].values,
                        data[nodes[j]].values,
                        z_data
                    )

                    if p > alpha:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        sep_sets[(i, j)] = cond_set
                        sep_sets[(j, i)] = cond_set
                        removed.append((i, j))
                        break

        if not removed:
            break
        depth += 1

    dag = {i: set() for i in range(n)}
    for i in range(n):
        for j in adj[i]:
            if j in adj[i] and i in adj[j]:
                is_constrained = False
                for (src, dst) in domain_constraints:
                    if nodes[j] == src and nodes[i] == dst:
                        is_constrained = True
                        break

                if is_constrained:
                    dag[j].add(i)
                elif (i, j) in sep_sets:
                    dag[i].add(j)
                elif (j, i) in sep_sets:
                    dag[j].add(i)
                else:
                    if abs(data[nodes[i]].corr(data[nodes[j]])) > 0:
                        dag[i].add(j)
            elif j in adj[i]:
                dag[i].add(j)

    return dag, nodes, sep_sets

domain_constraints = [
    (outcome_binary, 'cdr3_len'),
    (outcome_binary, 'positive_ratio'),
    (outcome_binary, 'aromatic_ratio'),
    (outcome_binary, 'glycine_ratio'),
    (outcome_binary, 'serine_ratio'),
    (outcome_binary, 'proline_count'),
    (outcome_binary, 'first_is_aromatic'),
    (outcome_binary, 'last_is_YH'),
    (outcome_continuous, 'cdr3_len'),
    (outcome_continuous, 'positive_ratio'),
    (outcome_continuous, 'aromatic_ratio'),
    (outcome_continuous, 'glycine_ratio'),
    (outcome_continuous, 'serine_ratio'),
    (outcome_continuous, 'proline_count'),
    (outcome_continuous, 'first_is_aromatic'),
    (outcome_continuous, 'last_is_YH'),
]

print('Running PC algorithm...')
dag, nodes, sep_sets = pc_algorithm(causal_data, alpha=0.01, domain_constraints=domain_constraints)

print('\nLearned DAG edges:')
for src in range(len(nodes)):
    for dst in dag[src]:
        print(f'  {nodes[src]} -> {nodes[dst]}')

fig, ax = plt.subplots(figsize=(14, 10))
pos = {}
layer1 = ['cdr3_len', 'positive_ratio', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'proline_count']
layer2 = ['first_is_aromatic', 'last_is_YH']
layer3 = ['rf2_passed', 'rf2_interaction_pae']
layer4 = ['backbone_id']

x_spacing = 2.0
for i, node in enumerate(layer1):
    pos[node] = (i * x_spacing, 2)
for i, node in enumerate(layer2):
    pos[node] = (i * x_spacing + 3, 1)
for i, node in enumerate(layer3):
    pos[node] = (i * x_spacing + 4, 0)
for i, node in enumerate(layer4):
    pos[node] = (0, 0.5)

node_idx = {name: idx for idx, name in enumerate(nodes)}

for node_name, (x, y) in pos.items():
    if node_name in nodes:
        color = '#e74c3c' if node_name in [outcome_binary, outcome_continuous] else \
                '#3498db' if node_name in treatment_vars_extended else '#95a5a6'
        ax.scatter(x, y, s=2000, c=color, zorder=5, alpha=0.8)
        ax.text(x, y, node_name, ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)

for src in range(len(nodes)):
    for dst in dag[src]:
        src_name = nodes[src]
        dst_name = nodes[dst]
        if src_name in pos and dst_name in pos:
            x1, y1 = pos[src_name]
            x2, y2 = pos[dst_name]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5, alpha=0.6))

ax.set_title('Learned Causal DAG (PC Algorithm with Domain Constraints)', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer2_causal_dag.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Causal DAG plot saved.')

print('\n--- 2.3 后门调整ATE估计 ---')

def backdoor_ate(data, treatment, outcome, confounders, binary_treatment=False):
    if binary_treatment:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict

        X = data[confounders].values
        T = data[treatment].values
        Y = data[outcome].values

        lr = LogisticRegression(max_iter=1000)
        ps = cross_val_predict(lr, X, T, cv=5, method='predict_proba')[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        ipw = np.where(T == 1, 1.0 / ps, 1.0 / (1 - ps))
        ipw = np.clip(ipw, 0, 10)

        ate = np.mean(Y * T * ipw - Y * (1 - T) * ipw) / np.mean(ipw)
        return ate, None
    else:
        from sklearn.linear_model import LinearRegression

        X_confounders = data[confounders].values
        T = data[treatment].values.reshape(-1, 1)
        Y = data[outcome].values

        X_full = np.hstack([T, X_confounders])
        lr = LinearRegression()
        lr.fit(X_full, Y)
        ate = lr.coef_[0]

        n = len(Y)
        k = X_full.shape[1]
        Y_pred = lr.predict(X_full)
        residuals = Y - Y_pred
        mse = np.sum(residuals ** 2) / (n - k - 1)
        XTX_inv = np.linalg.inv(X_full.T @ X_full)
        se = np.sqrt(mse * XTX_inv[0, 0])

        t_stat = ate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k - 1))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return ate, (se, t_stat, p_value, ci_lower, ci_upper)

ate_results = []

for tv in treatment_vars_extended:
    confounders = ['backbone_id']

    if tv in ['cdr3_len', 'proline_count']:
        median_val = feat_df[tv].median()
        feat_df[f'{tv}_binary'] = (feat_df[tv] > median_val).astype(int)
        binary_tv = f'{tv}_binary'
        ate, info = backdoor_ate(feat_df, binary_tv, outcome_binary, confounders, binary_treatment=True)
        se, t_stat, p_val, ci_l, ci_u = 0, 0, 0, 0, 0
        if info is not None:
            se, t_stat, p_val, ci_l, ci_u = info
        ate_results.append({
            'treatment': tv,
            'outcome': outcome_binary,
            'ATE': ate,
            'SE': se,
            't_stat': t_stat,
            'p_value': p_val,
            'CI_lower': ci_l,
            'CI_upper': ci_u,
            'type': 'binary_treatment',
        })

    ate_cont, info_cont = backdoor_ate(feat_df, tv, outcome_continuous, confounders, binary_treatment=False)
    if info_cont is not None:
        se, t_stat, p_val, ci_l, ci_u = info_cont
    else:
        se, t_stat, p_val, ci_l, ci_u = 0, 0, 0, 0, 0
    ate_results.append({
        'treatment': tv,
        'outcome': outcome_continuous,
        'ATE': ate_cont,
        'SE': se,
        't_stat': t_stat,
        'p_value': p_val,
        'CI_lower': ci_l,
        'CI_upper': ci_u,
        'type': 'continuous_treatment',
    })

ate_df = pd.DataFrame(ate_results)
ate_df.to_csv(os.path.join(OUTPUT_DIR, 'ate_estimates.csv'), index=False)

print('\n--- ATE估计结果 ---')
print(f'{"Treatment":25s} {"Outcome":25s} {"ATE":>8s} {"SE":>8s} {"t-stat":>8s} {"p-value":>10s} {"95% CI":>20s} {"Rule":>10s}')
print('-' * 120)

hard_rules = []
soft_rules = []

for _, row in ate_df.iterrows():
    sig = ''
    rule_type = ''

    if row['outcome'] == outcome_binary:
        if row['p_value'] < 0.05 and abs(row['ATE']) > 0.05:
            sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
            if abs(row['ATE']) > 0.20:
                rule_type = 'HARD'
                hard_rules.append(row)
            else:
                rule_type = 'SOFT'
                soft_rules.append(row)
    else:
        if row['p_value'] < 0.05 and abs(row['ATE']) > 0.5:
            sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
            if abs(row['ATE']) > 1.0:
                rule_type = 'HARD'
                hard_rules.append(row)
            else:
                rule_type = 'SOFT'
                soft_rules.append(row)

    ci_str = f"[{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]"
    print(f"{row['treatment']:25s} {row['outcome']:25s} {row['ATE']:8.4f} {row['SE']:8.4f} {row['t_stat']:8.3f} {row['p_value']:10.6f} {ci_str:>20s} {rule_type:>10s} {sig}")

print('\n--- 2.4 硬约束规则提取 ---')
print('\n基于ATE + Cox HR + 数据分布的硬约束:')

print('\n[HC-1] CDR3长度约束:')
len_pass = feat_df.groupby('cdr3_len').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
    final_cand=('final_candidate', 'sum'),
).reset_index()
len_pass['pass_rate'] = len_pass['rf2_pass'] / len_pass['total'] * 100
for _, row in len_pass.iterrows():
    marker = ' <<<' if row['pass_rate'] > 15 else ''
    print(f"  长度{int(row['cdr3_len'])}: 通过率={row['pass_rate']:.1f}%, 最终候选={int(row['final_cand'])}{marker}")

print('\n[HC-2] CDR3首残基约束:')
first_res = feat_df.groupby('first_residue').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
).reset_index()
first_res['rate'] = first_res['rf2_pass'] / first_res['total'] * 100
first_res = first_res.sort_values('rate', ascending=False)
for _, row in first_res.head(5).iterrows():
    marker = ' <<<' if row['rate'] > 50 else ''
    print(f"  {row['first_residue']}: 通过率={row['rate']:.1f}% (n={int(row['total'])}){marker}")

print('\n[HC-3] CDR3尾残基约束:')
last_res = feat_df.groupby('last_residue').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
).reset_index()
last_res['rate'] = last_res['rf2_pass'] / last_res['total'] * 100
last_res = last_res.sort_values('rate', ascending=False)
for _, row in last_res.head(5).iterrows():
    marker = ' <<<' if row['rate'] > 15 else ''
    print(f"  {row['last_residue']}: 通过率={row['rate']:.1f}% (n={int(row['total'])}){marker}")

print('\n[HC-4] 正电荷残基约束:')
feat_df['has_positive'] = (feat_df['positive_count'] > 0).astype(int)
pos_pass = feat_df.groupby('has_positive').agg(
    total=('rf2_passed', 'count'),
    rf2_pass=('rf2_passed', 'sum'),
).reset_index()
pos_pass['rate'] = pos_pass['rf2_pass'] / pos_pass['total'] * 100
for _, row in pos_pass.iterrows():
    label = '有正电荷' if row['has_positive'] == 1 else '无正电荷'
    print(f"  {label}: 通过率={row['rate']:.1f}% (n={int(row['total'])})")

print('\n--- 2.5 软偏好规则提取 ---')

print('\n[SP-1] 芳香族比例:')
for thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
    mask = feat_df['aromatic_ratio'] >= thresh
    if mask.sum() > 0:
        pass_rate = feat_df.loc[mask, 'rf2_passed'].mean() * 100
        print(f"  aromatic_ratio >= {thresh:.2f}: n={mask.sum()}, 通过率={pass_rate:.1f}%")

print('\n[SP-2] 甘氨酸比例上限:')
for thresh in [0.05, 0.10, 0.12, 0.15, 0.20]:
    mask = feat_df['glycine_ratio'] <= thresh
    if mask.sum() > 0:
        pass_rate = feat_df.loc[mask, 'rf2_passed'].mean() * 100
        print(f"  glycine_ratio <= {thresh:.2f}: n={mask.sum()}, 通过率={pass_rate:.1f}%")

print('\n[SP-3] 丝氨酸比例上限:')
for thresh in [0.05, 0.10, 0.15, 0.20]:
    mask = feat_df['serine_ratio'] <= thresh
    if mask.sum() > 0:
        pass_rate = feat_df.loc[mask, 'rf2_passed'].mean() * 100
        print(f"  serine_ratio <= {thresh:.2f}: n={mask.sum()}, 通过率={pass_rate:.1f}%")

print('\n--- 2.6 反模式检测 ---')

print('\n[AP-1] 连续甘氨酸 (GGG):')
has_ggg = feat_df['has_ggg']
if has_ggg.sum() > 0:
    print(f"  含GGG: n={has_ggg.sum()}, 通过率={feat_df.loc[has_ggg, 'rf2_passed'].mean()*100:.1f}%")
    print(f"  无GGG: n={(~has_ggg).sum()}, 通过率={feat_df.loc[~has_ggg, 'rf2_passed'].mean()*100:.1f}%")
else:
    print("  数据中未发现GGG模式")

print('\n[AP-2] 连续丝氨酸 (SSS):')
has_sss = feat_df['has_sss']
if has_sss.sum() > 0:
    print(f"  含SSS: n={has_sss.sum()}, 通过率={feat_df.loc[has_sss, 'rf2_passed'].mean()*100:.1f}%")
    print(f"  无SSS: n={(~has_sss).sum()}, 通过率={feat_df.loc[~has_sss, 'rf2_passed'].mean()*100:.1f}%")
else:
    print("  数据中未发现SSS模式")

print('\n[AP-3] 连续亮氨酸 (LL):')
has_ll = feat_df['has_ll']
if has_ll.sum() > 0:
    print(f"  含LL: n={has_ll.sum()}, 通过率={feat_df.loc[has_ll, 'rf2_passed'].mean()*100:.1f}%")
    print(f"  无LL: n={(~has_ll).sum()}, 通过率={feat_df.loc[~has_ll, 'rf2_passed'].mean()*100:.1f}%")
else:
    print("  数据中未发现LL模式")

print('\n[AP-4] CDR3过长(>=10):')
long_cdr3 = feat_df['cdr3_len'] >= 10
short_cdr3 = feat_df['cdr3_len'] <= 7
print(f"  CDR3>=10: n={long_cdr3.sum()}, 通过率={feat_df.loc[long_cdr3, 'rf2_passed'].mean()*100:.1f}%")
print(f"  CDR3<=7:  n={short_cdr3.sum()}, 通过率={feat_df.loc[short_cdr3, 'rf2_passed'].mean()*100:.1f}%")

print('\n--- 2.7 ATE森林图 ---')

fig, ax = plt.subplots(figsize=(12, 8))
binary_ate = ate_df[ate_df['outcome'] == outcome_binary].copy()
binary_ate = binary_ate.sort_values('ATE', ascending=True)

y_pos = range(len(binary_ate))
colors = ['#e74c3c' if row['p_value'] < 0.05 else '#95a5a6' for _, row in binary_ate.iterrows()]

ax.barh(y_pos, binary_ate['ATE'], color=colors, alpha=0.7, height=0.6)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(binary_ate['treatment'])
ax.set_xlabel('Average Treatment Effect (ATE) on RF2 Pass Rate')
ax.set_title('Causal ATE Estimates: Treatment Effects on RF2 Pass Rate', fontsize=13)

for i, (_, row) in enumerate(binary_ate.iterrows()):
    sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
    ax.text(row['ATE'], i, f" {row['ATE']:.4f}{sig}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer2_ate_forest.png'), dpi=150, bbox_inches='tight')
plt.close()
print('ATE forest plot saved.')

print('\n--- 2.8 因果效应热图 ---')

ate_pivot = ate_df[ate_df['outcome'] == outcome_continuous].copy()
ate_pivot = ate_pivot.set_index('treatment')['ATE']

fig, ax = plt.subplots(figsize=(8, 6))
ate_vals = ate_pivot.values.reshape(-1, 1)
sns.heatmap(ate_vals, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            yticklabels=ate_pivot.index, xticklabels=['interaction_pae'],
            ax=ax, cbar_kws={'label': 'ATE'})
ax.set_title('Causal Effect on RF2 Interaction PAE', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer2_ate_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('ATE heatmap saved.')

print('\n' + '=' * 70)
print('第二层分析完成！')
print('=' * 70)
