#!/usr/bin/env python3
"""
三项任务:
1. 提取21条anti_pattern序列并分析触发规则
2. 导出1928条序列为FASTA格式
3. v2.4 vs v2.4b氨基酸分布可视化对比
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent
OUTPUT = BASE / "output_v2.4_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ============================================================
# 任务1: 提取21条anti_pattern序列并分析触发规则
# ============================================================
def task1_anti_pattern_analysis():
    print("=" * 70)
    print("任务1: 提取anti_pattern序列并分析触发规则")
    print("=" * 70)

    log_df = pd.read_csv(BASE / "output_v2.4_test" / "filter_log_v2.4b.csv")
    anti = log_df[log_df['reject_reason'].str.startswith('anti_pattern')].copy()
    anti.insert(0, 'anti_id', [f'ANTI_{i+1:03d}' for i in range(len(anti))])

    print(f"\n  anti_pattern序列总数: {len(anti)}")

    # 按反模式类型分组
    for pattern_type in anti['reject_reason'].unique():
        sub = anti[anti['reject_reason'] == pattern_type]
        pattern_str = pattern_type.replace('anti_pattern_', '')
        print(f"\n  === 反模式: {pattern_str} ({len(sub)}条) ===")
        print(f"  触发规则: 序列中包含连续3个'{pattern_str[0]}'残基(即'{pattern_str}')")
        print(f"  规则依据: 含该模式的序列FC率 < 不含该模式序列FC率的30%")

        # 分析每条序列
        print(f"\n  {'ID':^12s} | {'序列':^10s} | {'首残基':^4s} | {'尾残基':^4s} | {'长度':^3s} | {'模式位置':^8s} | {'模式残基占比':^10s}")
        print(f"  {'-'*12} | {'-'*10} | {'-'*4} | {'-'*4} | {'-'*3} | {'-'*8} | {'-'*10}")

        for _, row in sub.iterrows():
            seq = row.get('cdr3', '?')
            if isinstance(seq, str) and len(seq) > 0:
                # 找到模式位置
                positions = []
                for i in range(len(seq) - len(pattern_str) + 1):
                    if seq[i:i+len(pattern_str)] == pattern_str:
                        positions.append(str(i))
                pos_str = ','.join(positions) if positions else '?'
                # 模式残基占比
                target_aa = pattern_str[0]
                aa_ratio = seq.count(target_aa) / len(seq)
                print(f"  {row['anti_id']:^12s} | {seq:^10s} | {seq[0]:^4s} | {seq[-1]:^4s} | {len(seq):^3d} | {pos_str:^8s} | {aa_ratio:>9.1%}")
            else:
                print(f"  {row['anti_id']:^12s} | {'?':^10s} | {'?':^4s} | {'?':^4s} | {'?':^3s} | {'?':^8s} | {'?':^10s}")

    # 汇总分析
    print(f"\n  触发规则汇总:")
    print(f"  ┌──────────┬──────────┬──────────────────────────────────────────────────┐")
    print(f"  │ 反模式   │ 数量     │ 触发规则详情                                     │")
    print(f"  ├──────────┼──────────┼──────────────────────────────────────────────────┤")

    pattern_details = {
        'GGG': '连续3个甘氨酸(Gly) → Cox HR=4.35(最强风险因子), 甘氨酸堆积导致CDR3柔性过高',
        'SSS': '连续3个丝氨酸(Ser) → Cox HR=1.77(风险因子), 丝氨酸堆积增加骨架极性',
        'LL':  '连续2个亮氨酸(Leu) → 疏水残基堆积导致空间位阻, CDR3构象不稳定',
    }
    for pattern_type in anti['reject_reason'].unique():
        pattern_str = pattern_type.replace('anti_pattern_', '')
        cnt = len(anti[anti['reject_reason'] == pattern_type])
        detail = pattern_details.get(pattern_str, '未知')
        print(f"  │ {pattern_str:^8s} │ {cnt:^8d} │ {detail:<48s} │")

    print(f"  └──────────┴──────────┴──────────────────────────────────────────────────┘")

    # 导出
    out_path = OUTPUT / "anti_pattern_sequences.csv"
    anti.to_csv(out_path, index=False)
    print(f"\n  导出: {out_path} ({len(anti)}条)")

    return anti


# ============================================================
# 任务2: 导出FASTA格式
# ============================================================
def task2_export_fasta():
    print("\n" + "=" * 70)
    print("任务2: 导出1928条序列为FASTA格式")
    print("=" * 70)

    df = pd.read_csv(BASE / "output_v2.4_test" / "generated_sequences_v2.4b.csv")
    fasta_path = OUTPUT / "generated_sequences_v2.4b.fasta"

    with open(fasta_path, 'w') as f:
        for i, row in df.iterrows():
            seq = row['cdr3']
            header = (f">CSC-O_v2.4b|seq_{i+1:04d}|len{row['cdr3_len']}"
                      f"|first_{seq[0]}|last_{seq[-1]}"
                      f"|soft_score_{row['soft_score']:.1f}"
                      f"|aro_{row['aromatic_ratio']:.2f}"
                      f"|gly_{row['glycine_ratio']:.2f}"
                      f"|hyd_{row['hydrophobic_ratio']:.2f}")
            f.write(header + "\n")
            f.write(seq + "\n")

    print(f"\n  FASTA文件已导出: {fasta_path}")
    print(f"  序列数: {len(df)}")
    print(f"  格式: 标准FASTA (兼容PyMOL / AlphaFold / Rosetta)")
    print(f"\n  FASTA header格式说明:")
    print(f"    >CSC-O_v2.4b|seq_XXXX|lenN|first_X|last_X|soft_score_X.X|aro_X.XX|gly_X.XX|hyd_X.XX")
    print(f"\n  示例 (前3条):")
    with open(fasta_path) as f:
        for _ in range(6):  # 3条序列 = 6行
            print(f"    {f.readline().strip()}")

    return fasta_path


# ============================================================
# 任务3: v2.4 vs v2.4b 氨基酸分布可视化对比
# ============================================================
def task3_visualization():
    print("\n" + "=" * 70)
    print("任务3: v2.4 vs v2.4b 氨基酸分布可视化对比")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 重新生成v2.4序列(原始参数)
    from csco_generator import load_strategy, generate_cdr3
    v24_strategy = load_strategy(str(BASE / "output_v2.4_test" / "design_strategy_v2.4.json"))
    rng24 = random.Random(42)
    np.random.seed(42)
    seqs24 = []
    for length in [6, 7]:
        seqs24.extend(generate_cdr3(v24_strategy, length, 1000, rng24, verbose=False))
    seqs24 = [s for s in seqs24 if s['soft_score'] >= 1.5]
    seen24 = set()
    unique24 = []
    for s in seqs24:
        if s['cdr3'] not in seen24:
            seen24.add(s['cdr3'])
            unique24.append(s)
    df24 = pd.DataFrame(unique24)

    df24b = pd.read_csv(BASE / "output_v2.4_analysis" / "generated_sequences_v2.4b.csv")

    def aa_dist(df):
        all_res = ''.join(df['cdr3'].tolist())
        total = len(all_res)
        counts = Counter(all_res)
        return {aa: counts.get(aa, 0) / total * 100 for aa in AMINO_ACIDS}

    dist24 = aa_dist(df24)
    dist24b = aa_dist(df24b)

    # Shannon熵
    def shannon(dist):
        probs = [v / 100 for v in dist.values() if v > 0]
        return -sum(p * np.log2(p) for p in probs)

    h24 = shannon(dist24)
    h24b = shannon(dist24b)

    # ---- 图1: 氨基酸分布对比柱状图 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: 并排柱状图
    ax = axes[0, 0]
    x = np.arange(len(AMINO_ACIDS))
    width = 0.35
    bars1 = ax.bar(x - width/2, [dist24[aa] for aa in AMINO_ACIDS], width,
                   label='v2.4', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x + width/2, [dist24b[aa] for aa in AMINO_ACIDS], width,
                   label='v2.4b', color='#DD8452', alpha=0.85)
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Amino Acid Distribution: v2.4 vs v2.4b', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(AMINO_ACIDS, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 1b: 差异柱状图
    ax = axes[0, 1]
    diffs = [dist24b[aa] - dist24[aa] for aa in AMINO_ACIDS]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in diffs]
    ax.bar(x, diffs, color=colors, alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Change (percentage points)', fontsize=12)
    ax.set_title('Distribution Change: v2.4b - v2.4', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(AMINO_ACIDS, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    # 标注显著变化
    for i, (aa, d) in enumerate(zip(AMINO_ACIDS, diffs)):
        if abs(d) > 0.5:
            ax.annotate(f'{d:+.1f}%', (i, d), textcoords="offset points",
                       xytext=(0, 5 if d > 0 else -12), ha='center', fontsize=8, fontweight='bold')

    # 1c: 首残基分布对比
    ax = axes[1, 0]
    first24 = df24['cdr3'].str[0].value_counts(normalize=True) * 100
    first24b = df24b['cdr3'].str[0].value_counts(normalize=True) * 100
    first_aas = ['F', 'W', 'Y']
    x_first = np.arange(len(first_aas))
    ax.bar(x_first - width/2, [first24.get(aa, 0) for aa in first_aas], width,
           label='v2.4', color='#4C72B0', alpha=0.85)
    ax.bar(x_first + width/2, [first24b.get(aa, 0) for aa in first_aas], width,
           label='v2.4b', color='#DD8452', alpha=0.85)
    ax.set_xlabel('First Residue', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('First Residue Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x_first)
    ax.set_xticklabels(first_aas, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    # 标注数值
    for i, aa in enumerate(first_aas):
        v24 = first24.get(aa, 0)
        v24b = first24b.get(aa, 0)
        ax.text(i - width/2, v24 + 0.5, f'{v24:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, v24b + 0.5, f'{v24b:.1f}%', ha='center', fontsize=9)

    # 1d: 关键特征雷达图
    ax = axes[1, 1]
    features = ['aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'hydrophobic_ratio']
    feature_labels = ['Aromatic', 'Glycine', 'Serine', 'Hydrophobic']
    v24_vals = [df24[f].mean() for f in features]
    v24b_vals = [df24b[f].mean() for f in features]

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    v24_vals_r = v24_vals + [v24_vals[0]]
    v24b_vals_r = v24b_vals + [v24b_vals[0]]
    angles_r = angles + [angles[0]]

    ax.plot(angles_r, v24_vals_r, 'o-', linewidth=2, label='v2.4', color='#4C72B0')
    ax.fill(angles_r, v24_vals_r, alpha=0.15, color='#4C72B0')
    ax.plot(angles_r, v24b_vals_r, 's-', linewidth=2, label='v2.4b', color='#DD8452')
    ax.fill(angles_r, v24b_vals_r, alpha=0.15, color='#DD8452')
    ax.set_xticks(angles)
    ax.set_xticklabels(feature_labels, fontsize=10)
    ax.set_title('Feature Profile Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加Shannon熵标注
    fig.text(0.5, 0.01,
             f'Shannon Entropy: v2.4 = {h24:.4f}  |  v2.4b = {h24b:.4f}  |  '
             f'Sequences: v2.4 = {len(df24)}  |  v2.4b = {len(df24b)}',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig_path = OUTPUT / "v24_vs_v24b_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  可视化图表已保存: {fig_path}")
    print(f"  包含4个子图:")
    print(f"    1) 氨基酸分布并排柱状图")
    print(f"    2) 分布差异柱状图(v2.4b - v2.4)")
    print(f"    3) 首残基F/W/Y分布对比")
    print(f"    4) 关键特征雷达图")

    # ---- 图2: 详细差异热图 ----
    fig2, ax2 = plt.subplots(figsize=(14, 5))

    # 构建数据矩阵: 行=氨基酸, 列=位置(首/中/尾/全局)
    positions = ['Global', 'First', 'Middle', 'Last']
    data = []
    for aa in AMINO_ACIDS:
        row_data = []
        # Global
        row_data.append(dist24b[aa] - dist24[aa])
        # First
        f24 = (df24['cdr3'].str[0] == aa).sum() / len(df24) * 100
        f24b = (df24b['cdr3'].str[0] == aa).sum() / len(df24b) * 100
        row_data.append(f24b - f24)
        # Middle
        mid24 = ''.join(s[1:-1] for s in df24['cdr3'])
        mid24b = ''.join(s[1:-1] for s in df24b['cdr3'])
        mid24_pct = mid24.count(aa) / len(mid24) * 100 if len(mid24) > 0 else 0
        mid24b_pct = mid24b.count(aa) / len(mid24b) * 100 if len(mid24b) > 0 else 0
        row_data.append(mid24b_pct - mid24_pct)
        # Last
        l24 = (df24['cdr3'].str[-1] == aa).sum() / len(df24) * 100
        l24b = (df24b['cdr3'].str[-1] == aa).sum() / len(df24b) * 100
        row_data.append(l24b - l24)
        data.append(row_data)

    data = np.array(data)
    im = ax2.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax2.set_xticks(range(len(positions)))
    ax2.set_xticklabels(positions, fontsize=11)
    ax2.set_yticks(range(len(AMINO_ACIDS)))
    ax2.set_yticklabels(AMINO_ACIDS, fontsize=10)
    ax2.set_title('Amino Acid Distribution Change Heatmap (v2.4b - v2.4, percentage points)',
                  fontsize=13, fontweight='bold')

    # 标注数值
    for i in range(len(AMINO_ACIDS)):
        for j in range(len(positions)):
            val = data[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax2.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax2, label='Change (pp)')
    plt.tight_layout()
    fig2_path = OUTPUT / "v24_vs_v24b_heatmap.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  热图已保存: {fig2_path}")

    # 打印关键差异摘要
    print(f"\n  关键差异摘要:")
    print(f"  {'AA':^4s} | {'全局变化':^10s} | {'首残基变化':^10s} | {'中间变化':^10s} | {'尾残基变化':^10s}")
    print(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for i, aa in enumerate(AMINO_ACIDS):
        if any(abs(data[i, j]) > 0.3 for j in range(4)):
            print(f"  {aa:^4s} | {data[i,0]:>+9.2f}% | {data[i,1]:>+9.2f}% | {data[i,2]:>+9.2f}% | {data[i,3]:>+9.2f}%")

    return fig_path, fig2_path


# ============================================================
# 主流程
# ============================================================
def main():
    task1_anti_pattern_analysis()
    task2_export_fasta()
    task3_visualization()

    print("\n" + "=" * 70)
    print("全部任务完成")
    print(f"输出目录: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
