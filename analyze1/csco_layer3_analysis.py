import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.insert(0, '.')
from csco_data_engineering import extract_cdr3_features

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))
embeddings = np.load(os.path.join(OUTPUT_DIR, 'esm2_embeddings.npy'))

print('=' * 70)
print('第三层：反事实序列导航')
print('=' * 70)

rf2_failed = feat_df[feat_df['rf2_passed'] == False].copy()
rf2_passed = feat_df[feat_df['rf2_passed'] == True].copy()

failed_idx = rf2_failed.index.values
passed_idx = rf2_passed.index.values

print(f'RF2失败序列数: {len(failed_idx)}')
print(f'RF2通过序列数: {len(passed_idx)}')

X_all = embeddings.copy()
Y_binary = feat_df['rf2_passed'].values.astype(int)
Y_pae = feat_df['rf2_interaction_pae'].values.astype(float)

# 构造有意义的二元 treatment 变量（避免 treatment=outcome 的无意义调用）
feat_df['first_is_aromatic'] = feat_df['first_residue'].isin(['Y', 'W', 'F']).astype(int)
feat_df['last_is_YH'] = feat_df['last_residue'].isin(['Y', 'H']).astype(int)
T_first_aromatic = feat_df['first_is_aromatic'].values

print('\n--- 3.2 Double ML: CATE估计 ---')

def double_ml_cate(X, T, Y, n_folds=5):
    n = len(Y)
    cate_estimates = np.zeros(n)
    t_stats = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        m_y = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            verbose=-1, random_state=42
        )
        m_y.fit(X_train, Y_train)
        Y_pred = m_y.predict(X_test)
        Y_resid = Y_test - Y_pred

        m_t = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            verbose=-1, random_state=42
        )
        m_t.fit(X_train, T_train)
        T_pred = m_t.predict_proba(X_test)[:, 1]
        T_resid = T_test - T_pred

        T_resid_sq_sum = np.sum(T_resid ** 2)
        if T_resid_sq_sum < 1e-8:
            cate_estimates[test_idx] = 0
            t_stats[test_idx] = 0
            continue

        theta = np.sum(T_resid * Y_resid) / T_resid_sq_sum
        cate_estimates[test_idx] = theta

        residuals = Y_resid - theta * T_resid
        se = np.sqrt(np.sum(residuals ** 2) / (n - 1) / T_resid_sq_sum)
        t_stats[test_idx] = theta / se if se > 0 else 0

    return cate_estimates, t_stats

print('Running Double ML for RF2 pass (binary outcome)...')
print('  Treatment: first_is_aromatic (CDR3 first residue is Y/W/F)')
cate_binary, tstat_binary = double_ml_cate(X_all, T_first_aromatic, Y_binary)
print(f'  Mean CATE: {cate_binary.mean():.4f}')
print(f'  Significant (|t|>2.0): {np.sum(np.abs(tstat_binary) > 2.0)} / {len(tstat_binary)}')

print('\nRunning Double ML for interaction PAE (continuous outcome)...')
cate_pae, tstat_pae = double_ml_cate(X_all, Y_binary, Y_pae)
print(f'  Mean CATE: {cate_pae.mean():.4f}')
print(f'  Significant (|t|>2.0): {np.sum(np.abs(tstat_pae) > 2.0)} / {len(tstat_pae)}')

np.save(os.path.join(OUTPUT_DIR, 'cate_binary.npy'), cate_binary)
np.save(os.path.join(OUTPUT_DIR, 'cate_pae.npy'), cate_pae)
np.save(os.path.join(OUTPUT_DIR, 'tstat_binary.npy'), tstat_binary)
np.save(os.path.join(OUTPUT_DIR, 'tstat_pae.npy'), tstat_pae)

print('\n--- 3.3 位置特异性CATE估计 ---')

pos_cate_results = []

for pos in range(13):
    mask_pos = feat_df['cdr3_len'] > pos
    if mask_pos.sum() < 100:
        continue

    pos_data = feat_df[mask_pos].copy()
    pos_idx = pos_data.index.values

    for aa in AMINO_ACIDS:
        aa_at_pos = pos_data['cdr3_sequence'].apply(
            lambda s: s[pos] if pd.notna(s) and len(s) > pos else 'X'
        )
        is_aa = (aa_at_pos == aa).astype(int).values

        if is_aa.sum() < 30:
            continue

        X_pos = embeddings[pos_idx]
        Y_pos = Y_pae[pos_idx]

        X_full = np.hstack([is_aa.reshape(-1, 1), X_pos])
        lr = Ridge(alpha=1.0)
        lr.fit(X_full, Y_pos)

        cate_val = lr.coef_[0]

        n = len(Y_pos)
        k = X_full.shape[1]
        Y_pred = lr.predict(X_full)
        residuals = Y_pos - Y_pred
        mse = np.sum(residuals ** 2) / (n - k - 1)
        XTX_inv = np.linalg.inv(X_full.T @ X_full)
        se = np.sqrt(mse * XTX_inv[0, 0])
        t_stat = cate_val / se if se > 0 else 0

        if abs(t_stat) > 2.0:
            pos_cate_results.append({
                'position': pos,
                'amino_acid': aa,
                'CATE': cate_val,
                'SE': se,
                't_stat': t_stat,
                'n_sequences': int(is_aa.sum()),
            })

pos_cate_df = pd.DataFrame(pos_cate_results)
if len(pos_cate_df) > 0:
    pos_cate_df = pos_cate_df.sort_values('CATE', ascending=True)
    pos_cate_df.to_csv(os.path.join(OUTPUT_DIR, 'position_specific_cate.csv'), index=False)
    print(f'\nPosition-specific CATE: {len(pos_cate_df)} significant results')
    print('\nTop 15 most beneficial mutations (lowest PAE):')
    print(pos_cate_df.head(15).to_string(index=False))
    print('\nTop 15 most harmful mutations (highest PAE):')
    print(pos_cate_df.tail(15).to_string(index=False))
else:
    print('No significant position-specific CATE results found.')

print('\n--- 3.4 成功模板最近邻映射 ---')

success_embeddings = embeddings[passed_idx]
fail_embeddings = embeddings[failed_idx]

nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(success_embeddings)

distances, nn_indices = nn.kneighbors(fail_embeddings)

print(f'Mapped {len(failed_idx)} failed sequences to 5 nearest successful templates')
print(f'Mean distance to nearest success: {distances[:, 0].mean():.4f}')
print(f'Median distance to nearest success: {np.median(distances[:, 0]):.4f}')

np.save(os.path.join(OUTPUT_DIR, 'nn_distances.npy'), distances)
np.save(os.path.join(OUTPUT_DIR, 'nn_indices.npy'), nn_indices)

print('\n--- 3.5 反事实编辑建议生成 ---')

counterfactual_suggestions = []

n_process = min(2000, len(failed_idx))
print(f'Generating suggestions for {n_process} failed sequences...')

for i in range(n_process):
    fail_row_idx = failed_idx[i]
    fail_row = feat_df.iloc[fail_row_idx]
    cdr3 = fail_row['cdr3_sequence']
    if pd.isna(cdr3) or len(cdr3) == 0:
        continue

    suggestions = []

    for pos in range(min(len(cdr3), 13)):
        original_aa = cdr3[pos]

        for aa in AMINO_ACIDS:
            if aa == original_aa:
                continue

            mutated = cdr3[:pos] + aa + cdr3[pos + 1:]
            new_feats = extract_cdr3_features(mutated)

            pae_change = 0.0
            if len(pos_cate_df) > 0:
                pos_aa_match = pos_cate_df[
                    (pos_cate_df['position'] == pos) & (pos_cate_df['amino_acid'] == aa)
                ]
                if len(pos_aa_match) > 0:
                    pae_change = pos_aa_match.iloc[0]['CATE']

            suggestions.append({
                'position': pos,
                'original': original_aa,
                'mutant': aa,
                'mutated_cdr3': mutated,
                'pae_change': pae_change,
                'new_aromatic_ratio': new_feats['aromatic_ratio'],
                'new_glycine_ratio': new_feats['glycine_ratio'],
                'new_serine_ratio': new_feats['serine_ratio'],
                'new_positive_count': new_feats['positive_count'],
            })

    suggestions.sort(key=lambda x: x['pae_change'])

    top3 = suggestions[:3]

    for rank, s in enumerate(top3):
        nn_idx = nn_indices[i, 0]
        success_row = feat_df.iloc[passed_idx[nn_idx]]
        success_cdr3 = str(success_row['cdr3_sequence'])
        edit_dist = sum(1 for a, b in zip(s['mutated_cdr3'], success_cdr3) if a != b) if len(s['mutated_cdr3']) == len(success_cdr3) else -1

        counterfactual_suggestions.append({
            'sequence_id': int(fail_row['global_sequence_index']),
            'original_cdr3': cdr3,
            'rank': rank + 1,
            'edit': f"Pos{s['position']} {s['original']}->{s['mutant']}",
            'mutated_cdr3': s['mutated_cdr3'],
            'predicted_pae_change': round(s['pae_change'], 2),
            'edit_distance_to_template': edit_dist,
            'nearest_success_cdr3': success_cdr3,
        })

cf_df = pd.DataFrame(counterfactual_suggestions)
cf_df.to_csv(os.path.join(OUTPUT_DIR, 'counterfactual_suggestions.csv'), index=False)
print(f'\nGenerated {len(cf_df)} counterfactual suggestions')

print('\n--- 3.6 高频挽救编辑模式聚类 ---')

if len(cf_df) > 0:
    edit_patterns = cf_df['edit'].value_counts()
    print('\nTop 20 most frequent edit patterns:')
    for pattern, count in edit_patterns.head(20).items():
        mean_pae = cf_df[cf_df['edit'] == pattern]['predicted_pae_change'].mean()
        print(f'  {pattern}: {count} occurrences, mean PAE change={mean_pae:.2f}')

print('\n--- 3.7 CDR3截短挽救分析 ---')

truncation_results = []
for i in range(min(2000, len(failed_idx))):
    fail_row_idx = failed_idx[i]
    fail_row = feat_df.iloc[fail_row_idx]
    cdr3 = fail_row['cdr3_sequence']
    if pd.isna(cdr3) or len(cdr3) <= 7:
        continue

    for target_len in [6, 7]:
        if len(cdr3) <= target_len:
            continue

        truncated = cdr3[:target_len]
        new_feats = extract_cdr3_features(truncated)

        # 从实际数据动态计算该长度的通过率（作为截短后预测通过率的依据）
        len_pass_rate = feat_df[feat_df['cdr3_len'] == target_len]['rf2_passed'].mean()
        predicted_rate = float(len_pass_rate) if not pd.isna(len_pass_rate) else 0.0

        truncation_results.append({
            'sequence_id': int(fail_row['global_sequence_index']),
            'original_cdr3': cdr3,
            'original_len': len(cdr3),
            'truncated_cdr3': truncated,
            'target_len': target_len,
            'predicted_pass_rate': predicted_rate,
            'new_aromatic_ratio': new_feats['aromatic_ratio'],
            'new_positive_count': new_feats['positive_count'],
        })

trunc_df = pd.DataFrame(truncation_results)
trunc_df.to_csv(os.path.join(OUTPUT_DIR, 'truncation_suggestions.csv'), index=False)
print(f'Generated {len(trunc_df)} truncation suggestions')

if len(trunc_df) > 0:
    len10_plus_7 = trunc_df[(trunc_df['original_len'] >= 10) & (trunc_df['target_len'] == 7)]
    print(f'CDR3>=10 truncation to 7: {len(len10_plus_7)} suggestions')

print('\n--- 3.8 可视化 ---')

if len(pos_cate_df) > 0:
    pivot_data = pos_cate_df.pivot_table(
        index='amino_acid', columns='position', values='CATE', aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_data, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
                ax=ax, cbar_kws={'label': 'CATE on interaction PAE'})
    ax.set_title('Position-Specific CATE: Mutation Effect on Interaction PAE', fontsize=13)
    ax.set_xlabel('CDR3 Position')
    ax.set_ylabel('Mutant Amino Acid')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_position_cate_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Position CATE heatmap saved.')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(distances[:, 0], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('Cosine Distance to Nearest Success Template')
axes[0].set_ylabel('Count')
axes[0].set_title('Failed→Success Template Distance', fontsize=12)
axes[0].axvline(x=np.median(distances[:, 0]), color='red', linestyle='--',
                label=f'Median={np.median(distances[:, 0]):.3f}')
axes[0].legend()

if len(cf_df) > 0:
    top_edits = edit_patterns.head(15)
    axes[1].barh(range(len(top_edits)), top_edits.values, color='#2ecc71', alpha=0.7)
    axes[1].set_yticks(range(len(top_edits)))
    axes[1].set_yticklabels(top_edits.index)
    axes[1].set_xlabel('Frequency')
    axes[1].set_title('Top 15 Most Frequent Salvage Edits', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_counterfactual_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Counterfactual analysis plots saved.')

print('\n--- 3.9 t-SNE嵌入空间可视化 ---')

sample_size = min(3000, len(embeddings))
np.random.seed(42)
sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
sample_embeds = embeddings[sample_idx]
sample_labels = Y_binary[sample_idx]

print(f'Running t-SNE on {sample_size} samples...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeds_2d = tsne.fit_transform(sample_embeds)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(embeds_2d[sample_labels == 0, 0], embeds_2d[sample_labels == 0, 1],
           c='#e74c3c', alpha=0.3, s=10, label='RF2 Failed')
ax.scatter(embeds_2d[sample_labels == 1, 0], embeds_2d[sample_labels == 1, 1],
           c='#2ecc71', alpha=0.6, s=20, label='RF2 Passed')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('ESM-2 Embedding Space: RF2 Pass vs Fail', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_tsne_embedding.png'), dpi=150, bbox_inches='tight')
plt.close()
print('t-SNE plot saved.')

print('\n--- 3.10 Double ML CATE分布可视化 ---')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(cate_pae, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('CATE on Interaction PAE')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of CATE Estimates (PAE outcome)', fontsize=12)
axes[0].axvline(x=0, color='red', linestyle='--')

sig_mask = np.abs(tstat_pae) > 2.0
axes[1].scatter(cate_pae[~sig_mask], tstat_pae[~sig_mask], c='#bdc3c7', alpha=0.3, s=5, label='Non-significant')
axes[1].scatter(cate_pae[sig_mask], tstat_pae[sig_mask], c='#e74c3c', alpha=0.5, s=10, label='Significant (|t|>2)')
axes[1].set_xlabel('CATE')
axes[1].set_ylabel('t-statistic')
axes[1].set_title('CATE vs t-statistic', fontsize=12)
axes[1].axhline(y=2, color='orange', linestyle='--', alpha=0.5)
axes[1].axhline(y=-2, color='orange', linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_cate_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('CATE distribution plot saved.')

print('\n' + '=' * 70)
print('第三层分析完成！')
print('=' * 70)
