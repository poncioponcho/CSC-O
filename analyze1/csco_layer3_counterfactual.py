import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import torch
import warnings
warnings.filterwarnings('ignore')
import os
import sys

OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))

print('=' * 70)
print('第三层：反事实序列导航')
print('=' * 70)

print('\n--- 3.1 ESM-2序列编码 ---')

ESM_EMBEDDING_PATH = os.path.join(OUTPUT_DIR, 'esm2_embeddings.npy')

if os.path.exists(ESM_EMBEDDING_PATH):
    print(f'Loading cached ESM-2 embeddings from {ESM_EMBEDDING_PATH}')
    embeddings = np.load(ESM_EMBEDDING_PATH)
    print(f'Embeddings shape: {embeddings.shape}')
else:
    print('Loading ESM-2 model (esm2_t12_35M_UR50D)...')
    import esm

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    sequences = feat_df['vh_sequence'].values
    n_seqs = len(sequences)
    embed_dim = 480
    embeddings = np.zeros((n_seqs, embed_dim), dtype=np.float32)

    batch_size = 8
    print(f'Encoding {n_seqs} sequences in batches of {batch_size}...')

    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        batch_seqs = [(f'seq_{i}', sequences[i]) for i in range(start, end)]

        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[12])
            token_representations = results['representations'][12]

        for i, (label, seq) in enumerate(batch_seqs):
            seq_len = len(seq)
            embeddings[start + i] = token_representations[i, 1:seq_len + 1].mean(dim=0).cpu().numpy()

        if (start // batch_size) % 100 == 0:
            print(f'  Processed {end}/{n_seqs} sequences...')

    np.save(ESM_EMBEDDING_PATH, embeddings)
    print(f'ESM-2 embeddings saved to {ESM_EMBEDDING_PATH}')
    print(f'Embeddings shape: {embeddings.shape}')

print('\n--- 3.2 构建处理矩阵 ---')

rf2_failed = feat_df[feat_df['rf2_passed'] == False].copy()
rf2_passed = feat_df[feat_df['rf2_passed'] == True].copy()

failed_idx = rf2_failed.index.values
passed_idx = rf2_passed.index.values

print(f'RF2失败序列数: {len(failed_idx)}')
print(f'RF2通过序列数: {len(passed_idx)}')

X_all = embeddings.copy()
Y_binary = feat_df['rf2_passed'].values.astype(int)
Y_pae = feat_df['rf2_interaction_pae'].values.astype(float)

print('\n--- 3.3 Double ML: CATE估计 ---')

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

        T_resid_var = np.var(T_resid)
        if T_resid_var < 1e-8:
            cate_estimates[test_idx] = 0
            t_stats[test_idx] = 0
            continue

        theta = np.sum(T_resid * Y_resid) / np.sum(T_resid ** 2)
        cate_estimates[test_idx] = theta

        residuals = Y_resid - theta * T_resid
        se = np.sqrt(np.sum(residuals ** 2) / (n - 1) / np.sum(T_resid ** 2))
        t_stats[test_idx] = theta / se if se > 0 else 0

    return cate_estimates, t_stats

print('Running Double ML for RF2 pass (binary outcome)...')
cate_binary, tstat_binary = double_ml_cate(X_all, Y_binary, Y_binary)
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

print('\n--- 3.4 位置特异性CATE估计（原子干预）---')

print('Computing position-specific mutation effects...')

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
        is_aa = (aa_at_pos == aa).astype(int)

        if is_aa.sum() < 30:
            continue

        X_pos = embeddings[pos_idx]
        Y_pos = Y_pae[pos_idx]

        from sklearn.linear_model import Ridge
        X_full = np.hstack([is_aa.values.reshape(-1, 1), X_pos])
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
                'n_sequences': is_aa.sum(),
            })

pos_cate_df = pd.DataFrame(pos_cate_results)
if len(pos_cate_df) > 0:
    pos_cate_df = pos_cate_df.sort_values('t_stat', ascending=True)
    pos_cate_df.to_csv(os.path.join(OUTPUT_DIR, 'position_specific_cate.csv'), index=False)
    print(f'\nPosition-specific CATE: {len(pos_cate_df)} significant results')
    print('\nTop 20 most beneficial mutations (lowest PAE):')
    print(pos_cate_df.head(20).to_string(index=False))
    print('\nTop 20 most harmful mutations (highest PAE):')
    print(pos_cate_df.tail(20).to_string(index=False))
else:
    print('No significant position-specific CATE results found.')

print('\n--- 3.5 成功模板最近邻映射 ---')

success_embeddings = embeddings[passed_idx]
fail_embeddings = embeddings[failed_idx]

nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(success_embeddings)

distances, indices = nn.kneighbors(fail_embeddings)

print(f'Mapped {len(failed_idx)} failed sequences to 5 nearest successful templates')
print(f'Mean distance to nearest success: {distances[:, 0].mean():.4f}')
print(f'Median distance to nearest success: {np.median(distances[:, 0]):.4f}')

np.save(os.path.join(OUTPUT_DIR, 'nn_distances.npy'), distances)
np.save(os.path.join(OUTPUT_DIR, 'nn_indices.npy'), indices)

print('\n--- 3.6 反事实编辑建议生成 ---')

counterfactual_suggestions = []

for i, fail_row_idx in enumerate(failed_idx[:2000]):
    fail_row = feat_df.iloc[fail_row_idx]
    cdr3 = fail_row['cdr3_sequence']
    if pd.isna(cdr3) or len(cdr3) == 0:
        continue

    suggestions = []

    for pos in range(len(cdr3)):
        original_aa = cdr3[pos]

        for aa in AMINO_ACIDS:
            if aa == original_aa:
                continue

            mutated = cdr3[:pos] + aa + cdr3[pos + 1:]

            from csco_data_engineering import extract_cdr3_features
            new_feats = extract_cdr3_features(mutated)

            pae_change = 0
            if pos < len(pos_cate_df):
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
                'new_cdr3_len': len(mutated),
                'new_aromatic_ratio': new_feats['aromatic_ratio'],
                'new_glycine_ratio': new_feats['glycine_ratio'],
                'new_serine_ratio': new_feats['serine_ratio'],
                'new_positive_count': new_feats['positive_count'],
            })

    suggestions.sort(key=lambda x: x['pae_change'])

    top3 = suggestions[:3]

    for rank, s in enumerate(top3):
        nn_idx = indices[i, 0]
        success_row = feat_df.iloc[passed_idx[nn_idx]]
        edit_dist = sum(1 for a, b in zip(s['mutated_cdr3'], success_row['cdr3_sequence']) if a != b) if len(s['mutated_cdr3']) == len(str(success_row['cdr3_sequence'])) else -1

        counterfactual_suggestions.append({
            'sequence_id': fail_row['global_sequence_index'],
            'original_cdr3': cdr3,
            'rank': rank + 1,
            'edit': f"Pos{s['position']} {s['original']}->{s['mutant']}",
            'mutated_cdr3': s['mutated_cdr3'],
            'predicted_pae_change': round(s['pae_change'], 2),
            'edit_distance_to_template': edit_dist,
            'nearest_success_cdr3': success_row['cdr3_sequence'],
        })

cf_df = pd.DataFrame(counterfactual_suggestions)
cf_df.to_csv(os.path.join(OUTPUT_DIR, 'counterfactual_suggestions.csv'), index=False)
print(f'\nGenerated {len(cf_df)} counterfactual suggestions for {len(failed_idx[:2000])} failed sequences')

print('\n--- 3.7 高频挽救编辑模式聚类 ---')

edit_patterns = cf_df['edit'].value_counts()
print('\nTop 20 most frequent edit patterns:')
for pattern, count in edit_patterns.head(20).items():
    mean_pae = cf_df[cf_df['edit'] == pattern]['predicted_pae_change'].mean()
    print(f'  {pattern}: {count} occurrences, mean PAE change={mean_pae:.2f}')

print('\n--- 3.8 CDR3截短挽救分析 ---')

truncation_results = []
for fail_row_idx in failed_idx[:2000]:
    fail_row = feat_df.iloc[fail_row_idx]
    cdr3 = fail_row['cdr3_sequence']
    if pd.isna(cdr3) or len(cdr3) <= 7:
        continue

    for target_len in [6, 7]:
        if len(cdr3) <= target_len:
            continue

        truncated = cdr3[:target_len]
        from csco_data_engineering import extract_cdr3_features
        new_feats = extract_cdr3_features(truncated)

        len6_pass_rate = 0.226
        len7_pass_rate = 0.516
        predicted_rate = len6_pass_rate if target_len == 6 else len7_pass_rate

        truncation_results.append({
            'sequence_id': fail_row['global_sequence_index'],
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
print(f'\nGenerated {len(trunc_df)} truncation suggestions')

if len(trunc_df) > 0:
    len10_plus = trunc_df[trunc_df['original_len'] >= 10]
    print(f'CDR3>=10 sequences with truncation to 7: {len(len10_plus[len10_plus["target_len"]==7])} suggestions')
    print(f'  Predicted pass rate after truncation to 7: ~51.6%')

print('\n--- 3.9 反事实编辑可视化 ---')

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
axes[0].set_title('Distribution: Failed→Success Template Distance', fontsize=12)
axes[0].axvline(x=np.median(distances[:, 0]), color='red', linestyle='--', label=f'Median={np.median(distances[:, 0]):.3f}')
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

print('\n--- 3.10 ESM-2嵌入空间t-SNE可视化 ---')

from sklearn.manifold import TSNE

sample_size = min(3000, len(embeddings))
sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
sample_embeds = embeddings[sample_idx]
sample_labels = Y_binary[sample_idx]

print(f'Running t-SNE on {sample_size} samples...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeds_2d = tsne.fit_transform(sample_embeds)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(embeds_2d[sample_labels == 0, 0], embeds_2d[sample_labels == 0, 1],
                     c='#e74c3c', alpha=0.3, s=10, label='RF2 Failed')
scatter = ax.scatter(embeds_2d[sample_labels == 1, 0], embeds_2d[sample_labels == 1, 1],
                     c='#2ecc71', alpha=0.6, s=20, label='RF2 Passed')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('ESM-2 Embedding Space: RF2 Pass vs Fail', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_tsne_embedding.png'), dpi=150, bbox_inches='tight')
plt.close()
print('t-SNE plot saved.')

print('\n' + '=' * 70)
print('第三层分析完成！')
print('=' * 70)
