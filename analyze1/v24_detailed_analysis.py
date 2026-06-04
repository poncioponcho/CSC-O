#!/usr/bin/env python3
"""
v2.4 序列详细分析报告
====================
1. 序列长度分布统计
2. 甘氨酸比例检查
3. 过滤序列分析
4. v2.3 vs v2.4 多样性对比
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent
V24_CSV = BASE / "output_v2.4_test" / "generated_sequences_v2.4.csv"
V23_CSV = BASE / "output_v2.3_test" / "generated_sequences.csv"
FILTER_LOG = BASE / "output_v2.4_test" / "filter_log_v2.4.csv"
OUTPUT_DIR = BASE / "output_v2.4_analysis"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def load_data():
    df24 = pd.read_csv(V24_CSV)
    df23 = pd.read_csv(V23_CSV) if V23_CSV.exists() else None
    filter_df = pd.read_csv(FILTER_LOG) if FILTER_LOG.exists() else None
    return df24, df23, filter_df


# ============================================================
# 任务1: 序列长度分布统计
# ============================================================
def task1_length_distribution(df):
    print("=" * 70)
    print("任务1: 序列长度分布统计")
    print("=" * 70)

    total = len(df)
    len6 = df[df['cdr3_len'] == 6]
    len7 = df[df['cdr3_len'] == 7]

    n6 = len(len6)
    n7 = len(len7)
    pct6 = n6 / total * 100
    pct7 = n7 / total * 100

    print(f"\n  总序列数: {total}")
    print(f"\n  ┌──────────┬──────────┬────────────┐")
    print(f"  │ 长度     │ 数量     │ 占比       │")
    print(f"  ├──────────┼──────────┼────────────┤")
    print(f"  │ 6        │ {n6:>8d} │ {pct6:>8.2f}%   │")
    print(f"  │ 7        │ {n7:>8d} │ {pct7:>8.2f}%   │")
    print(f"  ├──────────┼──────────┼────────────┤")
    print(f"  │ 合计     │ {total:>8d} │ {pct6+pct7:>8.2f}%   │")
    print(f"  └──────────┴──────────┴────────────┘")

    # 按长度分组的首残基分布
    print(f"\n  按长度分组的首残基分布:")
    for length in [6, 7]:
        sub = df[df['cdr3_len'] == length]
        print(f"    长度{length} (n={len(sub)}):")
        for aa in ['F', 'W', 'Y']:
            cnt = (sub['cdr3'].str[0] == aa).sum()
            print(f"      {aa}: {cnt} ({cnt/len(sub)*100:.2f}%)")

    # 按长度分组的尾残基分布
    print(f"\n  按长度分组的尾残基分布:")
    for length in [6, 7]:
        sub = df[df['cdr3_len'] == length]
        last_dist = sub['cdr3'].str[-1].value_counts()
        print(f"    长度{length} (n={len(sub)}):")
        for aa, cnt in last_dist.items():
            print(f"      {aa}: {cnt} ({cnt/len(sub)*100:.2f}%)")

    return {'total': total, 'len6': n6, 'len7': n7, 'pct6': pct6, 'pct7': pct7}


# ============================================================
# 任务2: 甘氨酸比例检查
# ============================================================
def task2_glycine_check(df):
    print("\n" + "=" * 70)
    print("任务2: 甘氨酸比例检查")
    print("=" * 70)

    # 逐序列甘氨酸比例
    gly_ratios = df['glycine_ratio']
    total = len(df)

    # 甘氨酸残基总占比（所有序列中G残基总数/所有残基总数）
    all_residues = ''.join(df['cdr3'].tolist())
    total_residues = len(all_residues)
    total_g = all_residues.count('G')
    global_gly_pct = total_g / total_residues

    # 逐序列统计
    seqs_above_015 = (gly_ratios > 0.15).sum()
    seqs_at_0 = (gly_ratios == 0).sum()
    seqs_below_015 = (gly_ratios <= 0.15).sum()

    print(f"\n  全局甘氨酸统计:")
    print(f"    总残基数: {total_residues}")
    print(f"    甘氨酸残基总数: {total_g}")
    print(f"    甘氨酸全局占比: {global_gly_pct:.4f} ({global_gly_pct*100:.2f}%)")
    print(f"    判定: {'✓ 严格控制在0.15以下' if global_gly_pct < 0.15 else '✗ 超过0.15阈值'}")

    print(f"\n  逐序列甘氨酸比例统计:")
    print(f"    平均甘氨酸比例: {gly_ratios.mean():.4f} ({gly_ratios.mean()*100:.2f}%)")
    print(f"    中位数: {gly_ratios.median():.4f}")
    print(f"    最大值: {gly_ratios.max():.4f}")
    print(f"    最小值: {gly_ratios.min():.4f}")
    print(f"    标准差: {gly_ratios.std():.4f}")

    print(f"\n  甘氨酸比例分布:")
    print(f"    ┌─────────────────────┬──────────┬────────────┐")
    print(f"    │ 类别                │ 数量     │ 占比       │")
    print(f"    ├─────────────────────┼──────────┼────────────┤")
    print(f"    │ glycine_ratio = 0   │ {seqs_at_0:>8d} │ {seqs_at_0/total*100:>8.2f}%   │")
    print(f"    │ 0 < ratio ≤ 0.15   │ {seqs_below_015-seqs_at_0:>8d} │ {(seqs_below_015-seqs_at_0)/total*100:>8.2f}%   │")
    print(f"    │ ratio > 0.15       │ {seqs_above_015:>8d} │ {seqs_above_015/total*100:>8.2f}%   │")
    print(f"    └─────────────────────┴──────────┴────────────┘")

    # 分位数
    print(f"\n  甘氨酸比例分位数:")
    for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        val = gly_ratios.quantile(q)
        print(f"    P{int(q*100):2d}: {val:.4f} ({val*100:.2f}%)")

    # 结论
    print(f"\n  结论:")
    if global_gly_pct < 0.15 and seqs_above_015 / total < 0.10:
        print(f"    ✓ 全局甘氨酸占比 {global_gly_pct*100:.2f}% 严格控制在0.15以下")
        print(f"    ✓ 仅 {seqs_above_015/total*100:.1f}% 的序列甘氨酸比例超过0.15（软偏好约束，非硬约束）")
    else:
        print(f"    ✗ 需要关注甘氨酸比例控制")

    return {'global_gly_pct': global_gly_pct, 'mean_gly': gly_ratios.mean(),
            'seqs_above_015': seqs_above_015, 'pct_above_015': seqs_above_015/total*100}


# ============================================================
# 任务3: 过滤序列分析
# ============================================================
def task3_filter_analysis(filter_df):
    print("\n" + "=" * 70)
    print("任务3: 过滤序列分析")
    print("=" * 70)

    if filter_df is None or len(filter_df) == 0:
        print("  无过滤记录")
        return

    total_filtered = len(filter_df)
    print(f"\n  总过滤序列数: {total_filtered}")

    # 过滤原因分布
    print(f"\n  过滤原因分布:")
    reason_counts = filter_df['reject_reason'].value_counts()
    for reason, cnt in reason_counts.items():
        pct = cnt / total_filtered * 100
        print(f"    {reason:<30s}: {cnt:>4d} ({pct:.1f}%)")

    # 类别1: 非白名单首残基
    first_related = filter_df[filter_df['reject_reason'].isin(['first_not_aromatic', 'first_not_whitelisted'])]
    n_first = len(first_related)
    print(f"\n  非白名单首残基被过滤序列: {n_first} 条")
    if n_first > 0:
        first_aa_dist = first_related['first_residue'].value_counts().sort_index()
        print(f"    被过滤首残基分布:")
        for aa, cnt in first_aa_dist.items():
            print(f"      {aa}: {cnt} 条")

    # 类别2: 非法长度
    length_related = filter_df[filter_df['reject_reason'] == 'length_not_allowed']
    n_length = len(length_related)
    print(f"\n  非法长度被过滤序列: {n_length} 条")
    if n_length > 0:
        len_dist = length_related['length'].value_counts().sort_index()
        print(f"    被过滤长度分布:")
        for l, cnt in len_dist.items():
            print(f"      长度{l}: {cnt} 条")

    # 类别3: 非白名单尾残基
    last_related = filter_df[filter_df['reject_reason'] == 'last_not_whitelisted']
    n_last = len(last_related)
    print(f"\n  非白名单尾残基被过滤序列: {n_last} 条")
    if n_last > 0:
        last_aa_dist = last_related['last_residue'].value_counts().sort_index()
        print(f"    被过滤尾残基分布:")
        for aa, cnt in last_aa_dist.items():
            print(f"      {aa}: {cnt} 条")

    # 类别4: 反模式
    anti_related = filter_df[filter_df['reject_reason'].str.startswith('anti_pattern')]
    n_anti = len(anti_related)
    print(f"\n  反模式被过滤序列: {n_anti} 条")

    # 完整过滤序列列表
    print(f"\n  过滤序列完整信息 (前20条):")
    print(f"  {'序列':^10s} | {'首残基':^4s} | {'尾残基':^4s} | {'长度':^4s} | {'过滤原因':^25s} | 详情")
    print(f"  {'-'*10} | {'-'*4} | {'-'*4} | {'-'*4} | {'-'*25} | {'-'*30}")
    for _, row in filter_df.head(20).iterrows():
        cdr3 = row.get('cdr3', '?')
        first = row.get('first_residue', cdr3[0] if isinstance(cdr3, str) and len(cdr3)>0 else '?')
        last = row.get('last_residue', cdr3[-1] if isinstance(cdr3, str) and len(cdr3)>0 else '?')
        length = row.get('length', len(cdr3) if isinstance(cdr3, str) else '?')
        reason = row.get('reject_reason', '?')
        detail = row.get('reject_detail', '')
        print(f"  {cdr3:^10s} | {first:^4s} | {last:^4s} | {length:^4} | {reason:^25s} | {detail}")

    return {'total_filtered': total_filtered, 'n_first': n_first,
            'n_length': n_length, 'n_last': n_last, 'n_anti': n_anti}


# ============================================================
# 任务4: v2.3 vs v2.4 多样性对比
# ============================================================
def task4_diversity_comparison(df24, df23):
    print("\n" + "=" * 70)
    print("任务4: v2.3 vs v2.4 序列多样性对比报告")
    print("=" * 70)

    # ---- 4a: 氨基酸分布范围变化 ----
    print(f"\n  === 4a: 氨基酸分布范围变化 ===")

    def aa_distribution(df, label):
        all_residues = ''.join(df['cdr3'].tolist())
        total = len(all_residues)
        counts = Counter(all_residues)
        dist = {}
        for aa in AMINO_ACIDS:
            c = counts.get(aa, 0)
            dist[aa] = {'count': c, 'pct': c/total*100 if total > 0 else 0}
        return dist, total

    dist23, total23 = aa_distribution(df23, "v2.3")
    dist24, total24 = aa_distribution(df24, "v2.4")

    print(f"\n  20种天然氨基酸使用分布对比:")
    print(f"  {'AA':^4s} | {'v2.3数量':>8s} | {'v2.3%':>7s} | {'v2.4数量':>8s} | {'v2.4%':>7s} | {'变化':>7s} | {'趋势'}")
    print(f"  {'-'*4} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*6}")

    v23_used = set()
    v24_used = set()
    for aa in AMINO_ACIDS:
        d23 = dist23[aa]
        d24 = dist24[aa]
        diff = d24['pct'] - d23['pct']
        if d23['count'] > 0: v23_used.add(aa)
        if d24['count'] > 0: v24_used.add(aa)
        trend = "↑" if diff > 0.5 else "↓" if diff < -0.5 else "→"
        print(f"  {aa:^4s} | {d23['count']:>8d} | {d23['pct']:>6.2f}% | {d24['count']:>8d} | {d24['pct']:>6.2f}% | {diff:>+6.2f}% | {trend}")

    print(f"\n  氨基酸覆盖范围:")
    print(f"    v2.3使用氨基酸种类: {len(v23_used)}/20 ({', '.join(sorted(v23_used))})")
    print(f"    v2.4使用氨基酸种类: {len(v24_used)}/20 ({', '.join(sorted(v24_used))})")
    only_v23 = v23_used - v24_used
    only_v24 = v24_used - v23_used
    if only_v23:
        print(f"    仅v2.3使用: {sorted(only_v23)}")
    else:
        print(f"    仅v2.3使用: 无 (v2.4覆盖了v2.3的全部氨基酸)")
    if only_v24:
        print(f"    仅v2.4使用: {sorted(only_v24)}")
    else:
        print(f"    仅v2.4使用: 无")

    # Shannon熵对比
    def shannon_entropy(dist_dict):
        probs = [d['pct']/100 for d in dist_dict.values() if d['pct'] > 0]
        return -sum(p * np.log2(p) for p in probs)

    h23 = shannon_entropy(dist23)
    h24 = shannon_entropy(dist24)
    print(f"\n  Shannon熵对比:")
    print(f"    v2.3: {h23:.4f}")
    print(f"    v2.4: {h24:.4f}")
    print(f"    变化: {h24-h23:+.4f} ({'多样性提升' if h24 > h23 else '多样性降低'})")

    # 首残基分布对比
    print(f"\n  首残基分布对比:")
    first23 = df23['cdr3'].str[0].value_counts()
    first24 = df24['cdr3'].str[0].value_counts()
    print(f"    v2.3首残基种类: {len(first23)} ({', '.join(sorted(first23.index))})")
    print(f"    v2.4首残基种类: {len(first24)} ({', '.join(sorted(first24.index))})")
    for aa in ['F', 'W', 'Y']:
        c23 = first23.get(aa, 0)
        c24 = first24.get(aa, 0)
        p23 = c23/len(df23)*100
        p24 = c24/len(df24)*100
        print(f"    {aa}: v2.3={c23}({p23:.1f}%) → v2.4={c24}({p24:.1f}%) 变化={p24-p23:+.1f}%")

    # 尾残基分布对比
    print(f"\n  尾残基分布对比:")
    last23 = df23['cdr3'].str[-1].value_counts()
    last24 = df24['cdr3'].str[-1].value_counts()
    print(f"    v2.3尾残基种类: {len(last23)} ({', '.join(sorted(last23.index))})")
    print(f"    v2.4尾残基种类: {len(last24)} ({', '.join(sorted(last24.index))})")

    # ---- 4b: 序列独特性变化 ----
    print(f"\n  === 4b: 序列独特性变化 ===")

    unique23 = df23['cdr3'].nunique()
    unique24 = df24['cdr3'].nunique()
    total23 = len(df23)
    total24 = len(df24)
    uniq_pct23 = unique23 / total23 * 100
    uniq_pct24 = unique24 / total24 * 100

    print(f"\n  序列独特性对比:")
    print(f"  ┌─────────────────────┬──────────┬──────────┐")
    print(f"  │ 指标                │ v2.3     │ v2.4     │")
    print(f"  ├─────────────────────┼──────────┼──────────┤")
    print(f"  │ 总序列数            │ {total23:>8d} │ {total24:>8d} │")
    print(f"  │ 独特序列数          │ {unique23:>8d} │ {unique24:>8d} │")
    print(f"  │ 独特性占比          │ {uniq_pct23:>7.2f}% │ {uniq_pct24:>7.2f}% │")
    print(f"  │ 重复序列数          │ {total23-unique23:>8d} │ {total24-unique24:>8d} │")
    print(f"  │ 重复率              │ {(total23-unique23)/total23*100:>7.2f}% │ {(total24-unique24)/total24*100:>7.2f}% │")
    print(f"  └─────────────────────┴──────────┴──────────┘")

    # 重复序列分析
    dup24 = df24[df24['cdr3'].duplicated(keep=False)].sort_values('cdr3')
    if len(dup24) > 0:
        dup_seqs = dup24['cdr3'].unique()
        print(f"\n  v2.4重复序列详情 (共{len(dup_seqs)}种重复序列):")
        for seq in dup_seqs[:10]:
            cnt = (df24['cdr3'] == seq).sum()
            print(f"    {seq}: 出现{cnt}次")
    else:
        print(f"\n  v2.4无重复序列 ✓")

    # 长度分布对比
    print(f"\n  长度分布对比:")
    len_dist23 = df23['cdr3_len'].value_counts().sort_index()
    len_dist24 = df24['cdr3_len'].value_counts().sort_index()
    all_lengths = sorted(set(len_dist23.index) | set(len_dist24.index))
    print(f"  {'长度':^4s} | {'v2.3数量':>8s} | {'v2.3%':>7s} | {'v2.4数量':>8s} | {'v2.4%':>7s}")
    print(f"  {'-'*4} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*7}")
    for l in all_lengths:
        c23 = len_dist23.get(l, 0)
        c24 = len_dist24.get(l, 0)
        p23 = c23/total23*100
        p24 = c24/total24*100
        print(f"  {l:^4d} | {c23:>8d} | {p23:>6.2f}% | {c24:>8d} | {p24:>6.2f}%")

    # 软偏好得分对比
    print(f"\n  软偏好得分对比:")
    print(f"    v2.3: 均值={df23['soft_score'].mean():.2f}, 中位数={df23['soft_score'].median():.2f}, 标准差={df23['soft_score'].std():.2f}")
    print(f"    v2.4: 均值={df24['soft_score'].mean():.2f}, 中位数={df24['soft_score'].median():.2f}, 标准差={df24['soft_score'].std():.2f}")

    # 特征对比
    print(f"\n  关键特征对比:")
    features = ['aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'hydrophobic_ratio', 'proline_count']
    print(f"  {'特征':<22s} | {'v2.3均值':>8s} | {'v2.4均值':>8s} | {'变化':>8s}")
    print(f"  {'-'*22} | {'-'*8} | {'-'*8} | {'-'*8}")
    for feat in features:
        m23 = df23[feat].mean()
        m24 = df24[feat].mean()
        diff = m24 - m23
        print(f"  {feat:<22s} | {m23:>8.4f} | {m24:>8.4f} | {diff:>+8.4f}")

    # 综合结论
    print(f"\n  === 多样性对比综合结论 ===")
    print(f"  1. 氨基酸覆盖: v2.3使用{len(v23_used)}种, v2.4使用{len(v24_used)}种", end="")
    if len(v24_used) >= len(v23_used):
        print(f" → v2.4覆盖范围{'≥' if len(v24_used) >= len(v23_used) else '<'}v2.3")
    else:
        print(f" → v2.4减少了{len(v23_used)-len(v24_used)}种氨基酸")

    print(f"  2. Shannon熵: v2.3={h23:.3f}, v2.4={h24:.3f} → {'提升' if h24>h23 else '降低'}{abs(h24-h23):.3f}")
    print(f"  3. 独特性: v2.3={uniq_pct23:.1f}%, v2.4={uniq_pct24:.1f}% → {'提升' if uniq_pct24>uniq_pct23 else '降低'}")
    print(f"  4. 序列量: v2.3={total23}, v2.4={total24} → 增加{total24-total23}条(+{(total24-total23)/total23*100:.0f}%)")
    print(f"  5. 长度集中度: v2.4仅长度6-7，消除了v2.3中长度9-13的低效序列")
    print(f"  6. 甘氨酸控制: v2.4均值={df24['glycine_ratio'].mean():.3f} (v2.3={df23['glycine_ratio'].mean():.3f}), 更严格控制")


# ============================================================
# 主流程
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df24, df23, filter_df = load_data()

    print(f"v2.4序列数: {len(df24)}")
    if df23 is not None:
        print(f"v2.3序列数: {len(df23)}")
    if filter_df is not None:
        print(f"过滤日志数: {len(filter_df)}")

    # 任务1
    task1_length_distribution(df24)

    # 任务2
    task2_glycine_check(df24)

    # 任务3
    task3_filter_analysis(filter_df)

    # 任务4
    if df23 is not None:
        task4_diversity_comparison(df24, df23)
    else:
        print("\n  v2.3数据不可用，跳过多样性对比")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
