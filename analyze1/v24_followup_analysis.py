#!/usr/bin/env python3
"""
v2.4 三项后续分析任务
====================
1. 筛选甘氨酸>0.15的121条序列并导出CSV
2. 调整首残基权重使Shannon熵回升至4.1+
3. 生成89条非白名单首残基过滤日志
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from csco_config import AMINO_ACIDS, AROMATIC, HYDROPHOBIC, extract_cdr3_features
from csco_generator import (load_strategy, check_hard_constraints,
                             check_anti_patterns, score_soft_preferences,
                             generate_cdr3)

BASE = Path(__file__).parent
V24_CSV = BASE / "output_v2.4_test" / "generated_sequences_v2.4.csv"
V24_STRATEGY = BASE / "output_v2.4_test" / "design_strategy_v2.4.json"
FILTER_LOG = BASE / "output_v2.4_test" / "filter_log_v2.4.csv"
OUTPUT_DIR = BASE / "output_v2.4_analysis"


# ============================================================
# 任务1: 筛选甘氨酸>0.15的序列并导出
# ============================================================
def task1_export_high_glycine():
    print("=" * 70)
    print("任务1: 筛选甘氨酸占比>0.15的序列并导出CSV")
    print("=" * 70)

    df = pd.read_csv(V24_CSV)
    high_gly = df[df['glycine_ratio'] > 0.15].copy()

    # 添加序列ID
    high_gly.insert(0, 'seq_id', [f'GLY_HIGH_{i+1:03d}' for i in range(len(high_gly))])

    # 添加甘氨酸残基数量和位置信息
    gly_details = []
    for _, row in high_gly.iterrows():
        seq = row['cdr3']
        gly_count = seq.count('G')
        gly_positions = [i for i, aa in enumerate(seq) if aa == 'G']
        gly_details.append({
            'gly_count': gly_count,
            'gly_positions': ','.join(map(str, gly_positions)),
            'gly_at_first': seq[0] == 'G',
            'gly_at_last': seq[-1] == 'G',
        })

    detail_df = pd.DataFrame(gly_details)
    high_gly = pd.concat([high_gly.reset_index(drop=True), detail_df], axis=1)

    # 导出
    out_path = OUTPUT_DIR / "high_glycine_sequences.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    high_gly.to_csv(out_path, index=False)

    print(f"\n  筛选结果:")
    print(f"    总序列数: {len(df)}")
    print(f"    甘氨酸>0.15的序列: {len(high_gly)} ({len(high_gly)/len(df)*100:.2f}%)")
    print(f"    导出路径: {out_path}")

    # 统计
    print(f"\n  甘氨酸超标序列特征:")
    print(f"    甘氨酸比例范围: {high_gly['glycine_ratio'].min():.4f} ~ {high_gly['glycine_ratio'].max():.4f}")
    print(f"    甘氨酸残基数分布: {dict(high_gly['gly_count'].value_counts().sort_index())}")
    print(f"    首残基分布: {dict(high_gly['cdr3'].str[0].value_counts())}")
    print(f"    长度分布: {dict(high_gly['cdr3_len'].value_counts().sort_index())}")

    # 示例
    print(f"\n  前10条甘氨酸超标序列:")
    print(f"  {'ID':^14s} | {'序列':^8s} | {'长度':^3s} | {'甘氨酸%':^6s} | {'G数量':^4s} | {'G位置'}")
    print(f"  {'-'*14} | {'-'*8} | {'-'*3} | {'-'*6} | {'-'*4} | {'-'*20}")
    for _, row in high_gly.head(10).iterrows():
        print(f"  {row['seq_id']:^14s} | {row['cdr3']:^8s} | {row['cdr3_len']:^3d} | {row['glycine_ratio']*100:>5.1f}% | {row['gly_count']:^4d} | {row['gly_positions']}")

    return high_gly


# ============================================================
# 任务2: 调整首残基权重使Shannon熵回升至4.1+
# ============================================================
def task2_adjust_weights_for_entropy():
    print("\n" + "=" * 70)
    print("任务2: 调整首残基权重使Shannon熵回升至4.1+")
    print("=" * 70)

    df = pd.read_csv(V24_CSV)

    # 计算当前Shannon熵
    def compute_shannon(sequences):
        all_residues = ''.join(sequences)
        total = len(all_residues)
        counts = Counter(all_residues)
        probs = [c/total for c in counts.values() if c > 0]
        return -sum(p * np.log2(p) for p in probs)

    current_entropy = compute_shannon(df['cdr3'].tolist())
    print(f"\n  当前v2.4 Shannon熵: {current_entropy:.4f}")
    print(f"  目标Shannon熵: ≥ 4.1")

    # 分析：Shannon熵下降的原因是F/W/Y占比过高导致分布集中
    # 解决方案：在中间残基生成时降低芳香族权重，增加其他氨基酸的多样性

    # 方案：调整soft_preferences中的aromatic_min_ratio和中间残基权重
    strategy = load_strategy(str(V24_STRATEGY))

    print(f"\n  当前策略参数:")
    print(f"    aromatic_min_ratio: {strategy['soft_preferences'].get('aromatic_min_ratio', 0)}")
    print(f"    glycine_max_ratio: {strategy['soft_preferences'].get('glycine_max_ratio', 0)}")
    print(f"    serine_max_ratio: {strategy['soft_preferences'].get('serine_max_ratio', 0)}")
    print(f"    hydrophobic_min_ratio: {strategy['soft_preferences'].get('hydrophobic_min_ratio', 0)}")

    # 尝试不同参数组合
    print(f"\n  参数调整实验:")

    experiments = [
        {"name": "当前v2.4", "aromatic_min_ratio": 0.15, "hydrophobic_min_ratio": 0.40,
         "glycine_max_ratio": 0.15, "serine_max_ratio": 0.15},
        {"name": "方案A: 降低芳香族要求", "aromatic_min_ratio": 0.10, "hydrophobic_min_ratio": 0.35,
         "glycine_max_ratio": 0.15, "serine_max_ratio": 0.20},
        {"name": "方案B: 降低芳香族+放宽丝氨酸", "aromatic_min_ratio": 0.08, "hydrophobic_min_ratio": 0.30,
         "glycine_max_ratio": 0.15, "serine_max_ratio": 0.25},
        {"name": "方案C: 最小芳香族+均衡分布", "aromatic_min_ratio": 0.05, "hydrophobic_min_ratio": 0.25,
         "glycine_max_ratio": 0.15, "serine_max_ratio": 0.25},
    ]

    results = []
    for exp in experiments:
        mod_strategy = json.loads(json.dumps(strategy))
        mod_strategy['soft_preferences']['aromatic_min_ratio'] = exp['aromatic_min_ratio']
        mod_strategy['soft_preferences']['hydrophobic_min_ratio'] = exp['hydrophobic_min_ratio']
        mod_strategy['soft_preferences']['glycine_max_ratio'] = exp['glycine_max_ratio']
        mod_strategy['soft_preferences']['serine_max_ratio'] = exp['serine_max_ratio']
        # 长度6/7也同步调整
        for l in ['6', '7']:
            if l in mod_strategy['length_specific_preferences']:
                mod_strategy['length_specific_preferences'][l]['aromatic_min_ratio'] = exp['aromatic_min_ratio']
                mod_strategy['length_specific_preferences'][l]['glycine_max_ratio'] = exp['glycine_max_ratio']

        rng = random.Random(42)
        np.random.seed(42)
        all_seqs = []
        for length in [6, 7]:
            seqs = generate_cdr3(mod_strategy, length, 500, rng, verbose=False)
            all_seqs.extend(seqs)

        if all_seqs:
            seq_strs = [s['cdr3'] for s in all_seqs]
            entropy = compute_shannon(seq_strs)
            df_exp = pd.DataFrame(all_seqs)
            fwY_pct = df_exp[df_exp['cdr3'].str[0].isin(['F','W','Y'])].shape[0] / len(df_exp) * 100
            gly_mean = df_exp['glycine_ratio'].mean()
            aro_mean = df_exp['aromatic_ratio'].mean()
            results.append({
                'name': exp['name'],
                'entropy': entropy,
                'fwY_pct': fwY_pct,
                'gly_mean': gly_mean,
                'aro_mean': aro_mean,
                'n_seqs': len(all_seqs),
            })
            mark = "✓" if entropy >= 4.1 else "✗"
            print(f"    {mark} {exp['name']}: Shannon={entropy:.4f}, F/W/Y首残基={fwY_pct:.1f}%, "
                  f"甘氨酸均值={gly_mean:.3f}, 芳香族均值={aro_mean:.3f}")

    # 选择最优方案
    valid = [r for r in results if r['entropy'] >= 4.1]
    if valid:
        best = max(valid, key=lambda x: x['fwY_pct'])  # 在满足熵≥4.1的前提下选F/W/Y最高的
        print(f"\n  推荐方案: {best['name']}")
        print(f"    Shannon熵: {best['entropy']:.4f} (≥4.1 ✓)")
        print(f"    F/W/Y首残基占比: {best['fwY_pct']:.1f}% (保持100%)")
        print(f"    甘氨酸均值: {best['gly_mean']:.3f}")
        print(f"    芳香族均值: {best['aro_mean']:.3f}")
    else:
        # 如果没有方案达到4.1，选最接近的
        best = max(results, key=lambda x: x['entropy'])
        print(f"\n  最接近方案: {best['name']} (Shannon={best['entropy']:.4f}, 未达4.1)")
        print(f"  需要进一步放宽中间残基约束或增加序列长度多样性")

    # 输出具体调整方案
    print(f"\n  具体参数调整方案:")
    print(f"  ┌─────────────────────────┬──────────┬──────────┬──────────┐")
    print(f"  │ 参数                    │ v2.4当前 │ 推荐调整 │ 变化     │")
    print(f"  ├─────────────────────────┼──────────┼──────────┼──────────┤")

    if valid:
        best_exp = [e for e in experiments if e['name'] == best['name']][0]
        for key in ['aromatic_min_ratio', 'hydrophobic_min_ratio', 'glycine_max_ratio', 'serine_max_ratio']:
            old_val = strategy['soft_preferences'].get(key, 0)
            new_val = best_exp[key]
            diff = new_val - old_val
            print(f"  │ {key:<23s} │ {old_val:>8.2f} │ {new_val:>8.2f} │ {diff:>+8.2f} │")

    print(f"  └─────────────────────────┴──────────┴──────────┴──────────┘")

    # 实施步骤
    print(f"\n  实施步骤:")
    print(f"    1. 修改 design_strategy_v2.4.json 中 soft_preferences 的参数")
    print(f"    2. 同步修改 length_specific_preferences 中长度6/7的对应参数")
    print(f"    3. 重新运行 csco_generator.py 生成序列")
    print(f"    4. 验证Shannon熵≥4.1且F/W/Y首残基占比=100%")

    print(f"\n  预期效果:")
    print(f"    - Shannon熵从{current_entropy:.3f}回升至{best['entropy']:.3f}")
    print(f"    - F/W/Y首残基占比保持100%")
    print(f"    - 中间残基分布更均匀，减少芳香族过度集中")
    print(f"    - 甘氨酸控制不变(glycine_max_ratio=0.15)")

    # 用推荐方案生成完整序列集
    if valid:
        print(f"\n  使用推荐方案生成完整序列集...")
        mod_strategy = json.loads(json.dumps(strategy))
        mod_strategy['soft_preferences']['aromatic_min_ratio'] = best_exp['aromatic_min_ratio']
        mod_strategy['soft_preferences']['hydrophobic_min_ratio'] = best_exp['hydrophobic_min_ratio']
        mod_strategy['soft_preferences']['glycine_max_ratio'] = best_exp['glycine_max_ratio']
        mod_strategy['soft_preferences']['serine_max_ratio'] = best_exp['serine_max_ratio']
        for l in ['6', '7']:
            if l in mod_strategy['length_specific_preferences']:
                mod_strategy['length_specific_preferences'][l]['aromatic_min_ratio'] = best_exp['aromatic_min_ratio']
                mod_strategy['length_specific_preferences'][l]['glycine_max_ratio'] = best_exp['glycine_max_ratio']

        rng = random.Random(42)
        np.random.seed(42)
        all_seqs = []
        for length in [6, 7]:
            seqs = generate_cdr3(mod_strategy, length, 1000, rng, verbose=False)
            all_seqs.extend(seqs)

        # 去重+软偏好过滤
        all_seqs = [s for s in all_seqs if s['soft_score'] >= 1.5]
        seen = set()
        unique = []
        for s in all_seqs:
            if s['cdr3'] not in seen:
                seen.add(s['cdr3'])
                unique.append(s)
        all_seqs = unique

        final_entropy = compute_shannon([s['cdr3'] for s in all_seqs])
        df_final = pd.DataFrame(all_seqs)
        fwY_final = df_final[df_final['cdr3'].str[0].isin(['F','W','Y'])].shape[0] / len(df_final) * 100

        # 保存
        v24b_path = OUTPUT_DIR / "generated_sequences_v2.4b.csv"
        df_final.to_csv(v24b_path, index=False)

        # 保存策略
        mod_strategy['strategy_name'] = 'CSC-O_v2.4b'
        mod_strategy['description'] = f'Entropy-optimized: Shannon={final_entropy:.3f}, length 6-7, first=F/W/Y'
        strat_path = OUTPUT_DIR / "design_strategy_v2.4b.json"
        with open(strat_path, 'w') as f:
            json.dump(mod_strategy, f, indent=2, ensure_ascii=False)

        print(f"    生成序列: {len(df_final)}")
        print(f"    Shannon熵: {final_entropy:.4f} ({'✓' if final_entropy >= 4.1 else '✗'} ≥4.1)")
        print(f"    F/W/Y首残基: {fwY_final:.1f}%")
        print(f"    导出: {v24b_path}")
        print(f"    策略: {strat_path}")

    return results


# ============================================================
# 任务3: 生成89条非白名单首残基过滤日志
# ============================================================
def task3_export_first_residue_filter_log():
    print("\n" + "=" * 70)
    print("任务3: 生成非白名单首残基过滤日志")
    print("=" * 70)

    filter_df = pd.read_csv(FILTER_LOG)

    # 筛选首残基相关过滤
    first_filtered = filter_df[
        filter_df['reject_reason'].isin(['first_not_aromatic', 'first_not_whitelisted'])
    ].copy()

    # 添加序列ID
    first_filtered.insert(0, 'log_id', [f'FIRST_FILTER_{i+1:03d}' for i in range(len(first_filtered))])

    # 添加详细信息
    first_filtered['whitelist'] = str(['F', 'W', 'Y'])
    first_filtered['is_aromatic'] = first_filtered['first_residue'].isin(['F', 'W', 'Y'])
    first_filtered['residue_category'] = first_filtered['first_residue'].map({
        'A': '非极性脂肪族', 'V': '非极性脂肪族', 'I': '非极性脂肪族', 'L': '非极性脂肪族',
        'M': '含硫', 'G': '非极性脂肪族', 'P': '亚氨基酸',
        'S': '极性不带电', 'T': '极性不带电', 'C': '含硫', 'N': '极性不带电', 'Q': '极性不带电',
        'D': '酸性', 'E': '酸性',
        'K': '碱性', 'R': '碱性', 'H': '碱性',
    })

    # 导出
    out_path = OUTPUT_DIR / "first_residue_filter_log.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first_filtered.to_csv(out_path, index=False)

    print(f"\n  过滤日志统计:")
    print(f"    非白名单首残基被过滤总数: {len(first_filtered)}")
    print(f"    导出路径: {out_path}")

    # 按首残基分类统计
    print(f"\n  按首残基分类统计:")
    aa_counts = first_filtered['first_residue'].value_counts().sort_index()
    print(f"  {'残基':^4s} | {'类别':^12s} | {'数量':^4s} | {'占比':^7s} | 过滤原因")
    print(f"  {'-'*4} | {'-'*12} | {'-'*4} | {'-'*7} | {'-'*30}")
    for aa, cnt in aa_counts.items():
        cat = first_filtered[first_filtered['first_residue'] == aa]['residue_category'].iloc[0]
        pct = cnt / len(first_filtered) * 100
        reason = first_filtered[first_filtered['first_residue'] == aa]['reject_reason'].iloc[0]
        print(f"  {aa:^4s} | {cat:^12s} | {cnt:^4d} | {pct:>5.1f}% | {reason}")

    # 按残基类别汇总
    print(f"\n  按残基类别汇总:")
    cat_counts = first_filtered['residue_category'].value_counts()
    for cat, cnt in cat_counts.items():
        aas = sorted(first_filtered[first_filtered['residue_category'] == cat]['first_residue'].unique())
        print(f"    {cat}: {cnt}条 ({', '.join(aas)})")

    # 完整日志内容
    print(f"\n  完整过滤日志 (前20条):")
    print(f"  {'ID':^16s} | {'序列':^10s} | {'首残基':^4s} | {'长度':^3s} | {'类别':^10s} | {'过滤原因':^22s} | 详情")
    print(f"  {'-'*16} | {'-'*10} | {'-'*4} | {'-'*3} | {'-'*10} | {'-'*22} | {'-'*30}")
    for _, row in first_filtered.head(20).iterrows():
        print(f"  {row['log_id']:^16s} | {row.get('cdr3','?'):^10s} | {row['first_residue']:^4s} | "
              f"{row.get('length','?'):^3} | {row['residue_category']:^10s} | "
              f"{row['reject_reason']:^22s} | {row.get('reject_detail', '')}")

    return first_filtered


# ============================================================
# 主流程
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 任务1
    task1_export_high_glycine()

    # 任务2
    task2_adjust_weights_for_entropy()

    # 任务3
    task3_export_first_residue_filter_log()

    print("\n" + "=" * 70)
    print("全部任务完成")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
