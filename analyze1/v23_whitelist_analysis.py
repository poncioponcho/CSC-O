#!/usr/bin/env python3
"""
v2.3 白名单分析综合脚本
任务1: 首残基分布统计报告
任务2: 白名单过滤日志导出
任务3: 尾残基白名单调整与多样性评估
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from csco_config import AMINO_ACIDS, AROMATIC, extract_cdr3_features
from csco_generator import (load_strategy, check_hard_constraints,
                             check_anti_patterns, score_soft_preferences)

STRATEGY_PATH = Path(__file__).parent / "output_server_v2.3" / "design_strategy.json"
OUTPUT_DIR = Path(__file__).parent / "output_v2.3_analysis"


def generate_with_log(strategy, n_samples=2000, seed=42):
    """生成序列并记录完整过滤日志"""
    rng = random.Random(seed)
    np.random.seed(seed)
    hc = strategy['hard_constraints']
    allowed_lengths = hc.get('cdr3_length_allowed', [5, 6, 7])
    preferred_lengths = hc.get('cdr3_length_preferred', [6, 7])
    length_weights_raw = strategy.get('length_generation_weights', {})

    all_generated = []
    filter_log = []

    if length_weights_raw:
        weight_sum = sum(length_weights_raw.get(str(l), 0.0) for l in allowed_lengths)
        if weight_sum > 0:
            for length in allowed_lengths:
                w = length_weights_raw.get(str(length), 0.0)
                n_target = max(int(n_samples * w / weight_sum), 100)
                seqs = _generate_cdr3_logged(strategy, length, n_target, rng, filter_log)
                all_generated.extend(seqs)
        else:
            length_weights_raw = {}
    if not length_weights_raw:
        n_per_length = n_samples // len(allowed_lengths)
        for length in allowed_lengths:
            n_target = n_per_length * 2 if length in preferred_lengths else n_per_length
            seqs = _generate_cdr3_logged(strategy, length, n_target, rng, filter_log)
            all_generated.extend(seqs)

    return all_generated, filter_log


def _generate_cdr3_logged(strategy, length, n_samples, rng, filter_log):
    """带完整日志的CDR3生成"""
    from csco_generator import generate_cdr3
    return generate_cdr3(strategy, length, n_samples, rng, verbose=False, filter_log=filter_log)


# ============================================================
# 任务1: 首残基分布统计报告
# ============================================================
def task1_first_residue_stats(df):
    print("=" * 70)
    print("任务1: 首残基分布统计报告")
    print("=" * 70)

    total = len(df)
    first_residues = df['cdr3'].str[0]
    counts = first_residues.value_counts().sort_index()

    print(f"\n总序列数: {total}")
    print(f"\n首残基详细分布:")
    print(f"  {'氨基酸':^6s} | {'数量':^8s} | {'占比':^10s} | {'条形图':^30s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*30}")

    for aa in ['F', 'W', 'Y']:
        cnt = (first_residues == aa).sum()
        pct = cnt / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {aa:^6s} | {cnt:^8d} | {pct:^9.2f}% | {bar}")

    print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*30}")
    whitelist_total = first_residues.isin(['F', 'W', 'Y']).sum()
    whitelist_pct = whitelist_total / total * 100
    print(f"  {'合计':^6s} | {whitelist_total:^8d} | {whitelist_pct:^9.2f}% |")

    # 非白名单残基检查
    non_wl = df[~first_residues.isin(['F', 'W', 'Y'])]
    if len(non_wl) > 0:
        print(f"\n  ⚠ 发现 {len(non_wl)} 条非白名单首残基序列:")
        for _, row in non_wl.head(10).iterrows():
            print(f"    {row['cdr3']} (首残基={row['cdr3'][0]})")
    else:
        print(f"\n  ✓ 所有 {total} 条序列首残基均在白名单 [F, W, Y] 内")

    # 按长度分组的首残基分布
    print(f"\n按CDR3长度分组的首残基分布:")
    for length in sorted(df['cdr3_len'].unique()):
        sub = df[df['cdr3_len'] == length]
        f_cnt = (sub['cdr3'].str[0] == 'F').sum()
        w_cnt = (sub['cdr3'].str[0] == 'W').sum()
        y_cnt = (sub['cdr3'].str[0] == 'Y').sum()
        n = len(sub)
        print(f"  长度{length:>2d}: F={f_cnt}({f_cnt/n*100:.1f}%) W={w_cnt}({w_cnt/n*100:.1f}%) Y={y_cnt}({y_cnt/n*100:.1f}%) 共{n}条")

    return {
        'total': total,
        'F_count': int((first_residues == 'F').sum()),
        'F_pct': round((first_residues == 'F').sum() / total * 100, 2),
        'W_count': int((first_residues == 'W').sum()),
        'W_pct': round((first_residues == 'W').sum() / total * 100, 2),
        'Y_count': int((first_residues == 'Y').sum()),
        'Y_pct': round((first_residues == 'Y').sum() / total * 100, 2),
    }


# ============================================================
# 任务2: 白名单过滤日志导出
# ============================================================
def task2_export_filter_log(filter_log, output_dir):
    print("\n" + "=" * 70)
    print("任务2: 白名单过滤日志导出")
    print("=" * 70)

    if not filter_log:
        print("  无过滤记录")
        return

    log_df = pd.DataFrame(filter_log)
    log_path = output_dir / "whitelist_filter_log.csv"
    log_df.to_csv(log_path, index=False)

    total_filtered = len(log_df)
    print(f"\n  过滤日志已导出: {log_path}")
    print(f"  总过滤记录数: {total_filtered}")

    # 按过滤原因分类统计
    print(f"\n  过滤原因分布:")
    reason_counts = log_df['reject_reason'].value_counts()
    for reason, cnt in reason_counts.items():
        pct = cnt / total_filtered * 100
        print(f"    {reason:<30s}: {cnt:>5d} ({pct:.1f}%)")

    # 首残基相关过滤详情
    first_related = log_df[log_df['reject_reason'].isin(['first_not_aromatic', 'first_not_whitelisted'])]
    if len(first_related) > 0:
        print(f"\n  首残基过滤详情 (共{len(first_related)}条):")
        first_aa_dist = first_related['first_residue'].value_counts().sort_index()
        for aa, cnt in first_aa_dist.items():
            print(f"    首残基={aa}: 被过滤 {cnt} 条")

    # 尾残基相关过滤详情
    last_related = log_df[log_df['reject_reason'] == 'last_not_whitelisted']
    if len(last_related) > 0:
        print(f"\n  尾残基过滤详情 (共{len(last_related)}条):")
        last_aa_dist = last_related['last_residue'].value_counts().sort_index()
        for aa, cnt in last_aa_dist.items():
            print(f"    尾残基={aa}: 被过滤 {cnt} 条")

    # 反模式过滤详情
    anti_related = log_df[log_df['reject_reason'].str.startswith('anti_pattern')]
    if len(anti_related) > 0:
        print(f"\n  反模式过滤详情 (共{len(anti_related)}条):")
        anti_dist = anti_related['reject_reason'].value_counts()
        for reason, cnt in anti_dist.items():
            print(f"    {reason}: {cnt} 条")

    # 导出前20条示例
    print(f"\n  过滤日志示例 (前10条):")
    for _, row in log_df.head(10).iterrows():
        print(f"    序列={row.get('cdr3', 'N/A'):15s} | 原因={row['reject_reason']:<25s} | 详情={row.get('reject_detail', 'N/A')}")

    return log_df


# ============================================================
# 任务3: 尾残基白名单调整与多样性评估
# ============================================================
def task3_last_residue_analysis(strategy, output_dir):
    print("\n" + "=" * 70)
    print("任务3: 尾残基白名单调整与多样性评估")
    print("=" * 70)

    hc = strategy['hard_constraints']
    current_last_wl = hc.get('cdr3_last_residue_whitelist', AMINO_ACIDS)
    print(f"\n  当前尾残基白名单: {current_last_wl} ({len(current_last_wl)}种)")

    # 分析当前v2.3生成数据中尾残基的实际分布
    csv_path = Path(__file__).parent / "output_v2.3_test" / "generated_sequences.csv"
    if csv_path.exists():
        df_current = pd.read_csv(csv_path)
    else:
        # 如果没有现成数据，重新生成
        print("  重新生成v2.3基准数据...")
        generated, _ = generate_with_log(strategy, n_samples=2000)
        df_current = pd.DataFrame(generated)

    current_last_dist = df_current['cdr3'].str[-1].value_counts().sort_index()
    print(f"\n  当前尾残基实际分布:")
    for aa in sorted(current_last_dist.index):
        cnt = current_last_dist[aa]
        pct = cnt / len(df_current) * 100
        in_wl = "✓" if aa in current_last_wl else "✗"
        print(f"    {in_wl} {aa}: {cnt:>4d} ({pct:.1f}%)")

    # 设计3种新尾残基白名单方案
    schemes = {
        "方案A_收紧至YH": ["Y", "H"],
        "方案B_芳香族+极性": ["Y", "H", "F", "W", "S", "T", "N"],
        "方案C_数据驱动Top6": _get_top_last_residues(df_current, 6),
    }

    print(f"\n  新尾残基白名单方案:")
    for name, wl in schemes.items():
        print(f"    {name}: {wl}")

    # 对每种方案生成序列并评估多样性
    results = {}
    for name, new_last_wl in schemes.items():
        modified_strategy = json.loads(json.dumps(strategy))
        modified_strategy['hard_constraints']['cdr3_last_residue_whitelist'] = new_last_wl

        generated, filt_log = generate_with_log(modified_strategy, n_samples=2000, seed=42)
        if not generated:
            results[name] = None
            continue

        df_new = pd.DataFrame(generated)
        metrics = _compute_diversity_metrics(df_new, name)
        metrics['last_whitelist'] = new_last_wl
        metrics['n_filtered'] = len(filt_log)
        metrics['filter_reasons'] = dict(Counter(r['reject_reason'] for r in filt_log)) if filt_log else {}
        results[name] = metrics

    # 基线多样性
    baseline_metrics = _compute_diversity_metrics(df_current, "v2.3_基线")

    # 对比表格
    print(f"\n  多样性对比评估:")
    print(f"  {'指标':<25s} | {'v2.3基线':>10s}", end="")
    for name in schemes:
        short = name.split("_")[0]
        print(f" | {short:>10s}", end="")
    print()
    print(f"  {'-'*25} | {'-'*10}", end="")
    for _ in schemes:
        print(f" | {'-'*10}", end="")
    print()

    metric_keys = ['n_sequences', 'unique_sequences', 'uniqueness_pct',
                   'n_unique_last_aa', 'last_aa_shannon', 'middle_aa_shannon',
                   'avg_soft_score']
    metric_labels = ['序列总数', '独特序列数', '独特性(%)', '尾残基种类数',
                     '尾残基Shannon熵', '中间残基Shannon熵', '平均软偏好得分']

    for key, label in zip(metric_keys, metric_labels):
        val = baseline_metrics.get(key, 0)
        if isinstance(val, float):
            print(f"  {label:<25s} | {val:>10.2f}", end="")
        else:
            print(f"  {label:<25s} | {val:>10}", end="")
        for name in schemes:
            r = results.get(name, {})
            v = r.get(key, 0)
            if isinstance(v, float):
                print(f" | {v:>10.2f}", end="")
            else:
                print(f" | {v:>10}", end="")
        print()

    # 过滤影响
    print(f"\n  过滤影响评估:")
    for name, r in results.items():
        if r is None:
            print(f"    {name}: 无法生成序列")
            continue
        print(f"    {name}: 白名单={r['last_whitelist']}, 被过滤={r['n_filtered']}条, 过滤原因={r['filter_reasons']}")

    # 调整建议
    print(f"\n  调整建议:")
    print(f"    当前v2.3尾残基白名单包含{len(current_last_wl)}种氨基酸, 覆盖面较广")
    print(f"    - 方案A(收紧至YH): 尾残基Shannon熵最低, 多样性损失最大, 但与ATE估计中last_is_YH=1.3一致")
    print(f"    - 方案B(芳香族+极性): 在保留结构稳定性的同时维持较高多样性")
    print(f"    - 方案C(数据驱动Top6): 基于实际生成频率选择, 平衡通过率与多样性")
    best = max(results.items(), key=lambda x: x[1]['last_aa_shannon'] if x[1] else 0)
    print(f"    综合推荐: {best[0]} (尾残基Shannon熵最高={best[1]['last_aa_shannon']:.3f})")

    return results


def _get_top_last_residues(df, top_n):
    """获取生成数据中出现频率最高的top_n个尾残基"""
    last_dist = df['cdr3'].str[-1].value_counts()
    return last_dist.head(top_n).index.tolist()


def _compute_diversity_metrics(df, label):
    """计算序列多样性指标"""
    total = len(df)
    unique_seqs = df['cdr3'].nunique()
    uniqueness_pct = unique_seqs / total * 100 if total > 0 else 0

    # 尾残基多样性
    last_aa_dist = df['cdr3'].str[-1].value_counts(normalize=True)
    n_unique_last = df['cdr3'].str[-1].nunique()
    last_shannon = -sum(p * np.log2(p) for p in last_aa_dist if p > 0)

    # 中间残基多样性 (去掉首尾)
    middle_aas = []
    for seq in df['cdr3']:
        if len(seq) > 2:
            middle_aas.extend(list(seq[1:-1]))
    if middle_aas:
        middle_dist = Counter(middle_aas)
        total_middle = sum(middle_dist.values())
        middle_shannon = -sum((c/total_middle) * np.log2(c/total_middle) for c in middle_dist.values() if c > 0)
    else:
        middle_shannon = 0

    # 平均软偏好得分
    avg_soft = df['soft_score'].mean() if 'soft_score' in df.columns else 0

    return {
        'n_sequences': total,
        'unique_sequences': unique_seqs,
        'uniqueness_pct': round(uniqueness_pct, 2),
        'n_unique_last_aa': n_unique_last,
        'last_aa_shannon': round(last_shannon, 3),
        'middle_aa_shannon': round(middle_shannon, 3),
        'avg_soft_score': round(avg_soft, 2),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = load_strategy(str(STRATEGY_PATH))

    # 生成带完整过滤日志的序列集
    print("正在生成序列 (含过滤日志)...")
    generated, filter_log = generate_with_log(strategy, n_samples=2000, seed=42)

    # 保存生成的序列
    df = pd.DataFrame(generated)
    csv_path = OUTPUT_DIR / "generated_sequences_v2.3.csv"
    df.to_csv(csv_path, index=False)
    print(f"已保存 {len(df)} 条序列至 {csv_path}")

    # 任务1
    stats = task1_first_residue_stats(df)

    # 保存统计报告
    stats_path = OUTPUT_DIR / "first_residue_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  统计报告已保存: {stats_path}")

    # 任务2
    log_df = task2_export_filter_log(filter_log, OUTPUT_DIR)

    # 任务3
    task3_results = task3_last_residue_analysis(strategy, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("所有任务完成")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
