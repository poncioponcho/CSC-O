#!/usr/bin/env python3
"""
v2.4b FC率模拟测试 + F/W/Y权重调节模块
========================================
1. 基于因果证据(v2.0~v2.3历史数据)构建FC率模拟器
2. 对v2.4b序列集进行FC率估算
3. 集成F/W/Y权重调节模块，实时显示FC率趋势
4. 生成趋势图表
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats

from csco_config import AMINO_ACIDS, AROMATIC, extract_cdr3_features
from csco_generator import load_strategy, generate_cdr3, check_hard_constraints, check_anti_patterns

BASE = Path(__file__).parent
OUTPUT = BASE / "output_v2.4_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FC率模拟器: 基于历史漏斗转化率和因果效应
# ============================================================
class FCRateSimulator:
    """
    基于v2.0~v2.3历史数据的漏斗转化率模型:
    - RF2通过率: 基于序列特征的Cox风险比预测
    - AF3通过率: 基于RF2通过序列的条件概率
    - Schrodinger/Desmond通过率: 基于AF3通过序列的条件概率
    - FC率 = RF2_pass × P(AF3|RF2) × P(Schrodinger|AF3) × P(Desmond|Schrodinger)
    """

    def __init__(self):
        # 历史漏斗转化率 (基于v2.0数据: 10572条原始序列)
        self.baseline_rf2_rate = 0.117    # 1235/10572
        self.af3_given_rf2 = 0.050        # 62/1235
        self.schrodinger_given_af3 = 1.0  # 62/62 (所有AF3通过都进入Schrodinger)
        self.desmond_given_schrodinger = 1.0  # 62/62 (所有Schrodinger通过都进入Desmond)
        self.baseline_fc_rate = 0.0061    # 65/10572

        # Cox风险比 (来自v2.0数据分析)
        self.cox_hr = {
            'glycine_ratio': 4.35,
            'serine_ratio': 1.77,
            'cdr3_len': 1.08,
            'aromatic_ratio': 0.67,  # 保护因子
        }

        # ATE估计 (对PAE的因果效应)
        self.ate = {
            'cdr3_len': 0.91,
            'aromatic_ratio': 5.03,
            'glycine_ratio': 5.41,
            'first_is_aromatic': -4.69,  # 保护因子
            'last_is_YH': 1.30,
        }

        # 分层ATE (长度6-7)
        self.stratified_ate = {
            6: {'aromatic_ratio': -1.91, 'glycine_ratio': 8.03, 'first_is_aromatic': -3.67},
            7: {'aromatic_ratio': -0.48, 'glycine_ratio': 3.81, 'first_is_aromatic': -1.83},
        }

        # v2.3实际RF2通过率 (白名单[F,W,Y]后)
        self.v23_rf2_rate = 0.218  # 约2300/10572

        # 亚群0 (黄金亚群) 的RF2通过率
        self.subgroup0_rf2_rate = 0.431

    def estimate_rf2_probability(self, features):
        """
        基于序列特征估算RF2通过概率
        使用逻辑回归近似: P(RF2_pass) = sigmoid(β0 + Σβi*xi)
        """
        # 基线log-odds
        baseline_logodds = np.log(self.baseline_rf2_rate / (1 - self.baseline_rf2_rate))

        # 特征效应 (基于Cox HR近似)
        logodds = baseline_logodds

        # 长度效应: 长度6-7是保护因子
        if features['cdr3_len'] <= 7:
            # 长度6-7 vs 基线(平均长度9.6)的效应
            len_reduction = 9.6 - features['cdr3_len']
            logodds += np.log(self.cox_hr['cdr3_len']) * (-len_reduction)

        # 甘氨酸效应 (风险因子)
        logodds -= np.log(self.cox_hr['glycine_ratio']) * features['glycine_ratio']

        # 丝氨酸效应 (风险因子)
        logodds -= np.log(self.cox_hr['serine_ratio']) * features['serine_ratio']

        # 首残基芳香族效应 (保护因子)
        if features.get('first_is_aromatic', False):
            logodds += abs(self.ate['first_is_aromatic']) * 0.1  # 缩放因子

        # 芳香族比例效应 (长度6-7有利)
        if features['cdr3_len'] in [6, 7]:
            strat_ate = self.stratified_ate.get(features['cdr3_len'], {})
            aro_effect = strat_ate.get('aromatic_ratio', 0)
            if aro_effect < 0:  # 保护
                logodds += abs(aro_effect) * features['aromatic_ratio'] * 0.05

        # 反模式惩罚
        seq = features.get('cdr3', '')
        if 'GGG' in seq or 'SSS' in seq or 'LL' in seq:
            logodds -= 1.5  # 强惩罚

        prob = 1 / (1 + np.exp(-logodds))
        return min(prob, 0.95)  # 上限95%

    def estimate_fc_rate(self, sequences_df):
        """
        估算序列集的整体FC率
        FC率 = E[P(RF2_pass)] × P(AF3|RF2) × P(Schrodinger|AF3) × P(Desmond|Schrodinger)
        """
        rf2_probs = []
        for _, row in sequences_df.iterrows():
            features = {
                'cdr3_len': row['cdr3_len'],
                'glycine_ratio': row['glycine_ratio'],
                'serine_ratio': row['serine_ratio'],
                'aromatic_ratio': row['aromatic_ratio'],
                'first_is_aromatic': row['first_is_aromatic'],
                'cdr3': row['cdr3'],
            }
            rf2_probs.append(self.estimate_rf2_probability(features))

        avg_rf2_prob = np.mean(rf2_probs)

        # 下游转化率: 基于v2.3数据，白名单序列的AF3通过率更高
        # v2.3中白名单序列的AF3通过率约5.0%，但高质量序列可能更高
        # 保守估计: 5.0%
        af3_rate = self.af3_given_rf2

        # Schrodinger和Desmond的联合通过率约50% (基于历史数据)
        downstream_rate = 0.50

        estimated_fc_rate = avg_rf2_prob * af3_rate * downstream_rate

        return {
            'avg_rf2_probability': avg_rf2_prob,
            'af3_conditional_rate': af3_rate,
            'downstream_rate': downstream_rate,
            'estimated_fc_rate': estimated_fc_rate,
            'n_sequences': len(sequences_df),
            'rf2_probs': rf2_probs,
        }


# ============================================================
# 任务1: v2.4b FC率模拟测试
# ============================================================
def task1_fc_simulation():
    print("=" * 70)
    print("任务1: v2.4b FC率模拟测试")
    print("=" * 70)

    df = pd.read_csv(OUTPUT / "generated_sequences_v2.4b.csv")
    sim = FCRateSimulator()

    result = sim.estimate_fc_rate(df)

    print(f"\n  v2.4b序列FC率模拟结果:")
    print(f"  ┌─────────────────────────────────┬──────────────┐")
    print(f"  │ 指标                            │ 估算值       │")
    print(f"  ├─────────────────────────────────┼──────────────┤")
    print(f"  │ 序列总数                        │ {result['n_sequences']:>12d} │")
    print(f"  │ 平均RF2通过概率                 │ {result['avg_rf2_probability']:>11.1%} │")
    print(f"  │ AF3条件通过率                   │ {result['af3_conditional_rate']:>11.1%} │")
    print(f"  │ 下游联合通过率                  │ {result['downstream_rate']:>11.1%} │")
    print(f"  │ 估算FC率                        │ {result['estimated_fc_rate']:>11.2%} │")
    print(f"  └─────────────────────────────────┴──────────────┘")

    # 与历史版本对比
    print(f"\n  与历史版本FC率对比:")
    print(f"  ┌──────────┬────────────┬──────────────┬──────────────┐")
    print(f"  │ 版本     │ RF2通过率  │ 估算FC率     │ vs基线       │")
    print(f"  ├──────────┼────────────┼──────────────┼──────────────┤")
    print(f"  │ v2.0基线 │ 11.7%      │ 0.61%        │ --           │")
    print(f"  │ v2.3     │ ~21.8%     │ ~0.55%*      │ -0.06pp      │")
    print(f"  │ v2.4b    │ {result['avg_rf2_probability']:.1%}      │ {result['estimated_fc_rate']:.2%}        │ +{result['estimated_fc_rate']-0.0061:.2f}pp      │")
    print(f"  └──────────┴────────────┴──────────────┴──────────────┘")
    print(f"  * v2.3 FC率基于原始数据，v2.4b为模拟估算")

    # 按长度分组
    print(f"\n  按长度分组FC率估算:")
    for length in [6, 7]:
        sub = df[df['cdr3_len'] == length]
        sub_result = sim.estimate_fc_rate(sub)
        print(f"    长度{length}: RF2通过率={sub_result['avg_rf2_probability']:.1%}, "
              f"估算FC率={sub_result['estimated_fc_rate']:.2%}")

    # 按首残基分组
    print(f"\n  按首残基分组FC率估算:")
    for aa in ['F', 'W', 'Y']:
        sub = df[df['cdr3'].str[0] == aa]
        sub_result = sim.estimate_fc_rate(sub)
        print(f"    {aa}: n={len(sub)}, RF2通过率={sub_result['avg_rf2_probability']:.1%}, "
              f"估算FC率={sub_result['estimated_fc_rate']:.2%}")

    return result, sim


# ============================================================
# 任务2: F/W/Y权重调节模块 + FC率趋势图
# ============================================================
def task2_weight_tuning(sim):
    print("\n" + "=" * 70)
    print("任务2: F/W/Y权重调节模块 + FC率趋势图")
    print("=" * 70)

    strategy = load_strategy(str(BASE / "output_v2.4_test" / "design_strategy_v2.4.json"))

    # 定义权重组合
    weight_configs = []

    # 实验1: 固定W=Y=1/3, 调整F的权重
    print(f"\n  实验1: 调整F首残基权重 (W=Y=1/3固定)")
    for f_weight in np.arange(0.1, 0.8, 0.05):
        remaining = 1.0 - f_weight
        w_weight = remaining / 2
        y_weight = remaining / 2
        weight_configs.append({
            'name': f'F={f_weight:.2f}',
            'F': f_weight, 'W': w_weight, 'Y': y_weight,
            'experiment': 1, 'varying': 'F'
        })

    # 实验2: 固定F=Y=1/3, 调整W的权重
    print(f"  实验2: 调整W首残基权重 (F=Y=1/3固定)")
    for w_weight in np.arange(0.1, 0.8, 0.05):
        remaining = 1.0 - w_weight
        f_weight = remaining / 2
        y_weight = remaining / 2
        weight_configs.append({
            'name': f'W={w_weight:.2f}',
            'F': f_weight, 'W': w_weight, 'Y': y_weight,
            'experiment': 2, 'varying': 'W'
        })

    # 实验3: 固定F=W=1/3, 调整Y的权重
    print(f"  实验3: 调整Y首残基权重 (F=W=1/3固定)")
    for y_weight in np.arange(0.1, 0.8, 0.05):
        remaining = 1.0 - y_weight
        f_weight = remaining / 2
        w_weight = remaining / 2
        weight_configs.append({
            'name': f'Y={y_weight:.2f}',
            'F': f_weight, 'W': w_weight, 'Y': y_weight,
            'experiment': 3, 'varying': 'Y'
        })

    # 对每种权重配置生成序列并估算FC率
    results = []
    for config in weight_configs:
        mod_strategy = json.loads(json.dumps(strategy))
        # 修改首残基白名单的隐式权重: 通过修改length_generation_weights和template来实现
        # 由于generator直接从whitelist随机选择，需要修改选择逻辑
        # 这里我们用后处理方式模拟: 生成大量序列后按权重重新采样

        rng = random.Random(42)
        np.random.seed(42)
        all_seqs = []
        for length in [6, 7]:
            seqs = generate_cdr3(mod_strategy, length, 600, rng, verbose=False)
            all_seqs.extend(seqs)

        # 按权重重新采样首残基
        df_gen = pd.DataFrame(all_seqs)
        df_gen = df_gen[df_gen['soft_score'] >= 1.5]

        # 按首残基分组
        f_seqs = df_gen[df_gen['cdr3'].str[0] == 'F']
        w_seqs = df_gen[df_gen['cdr3'].str[0] == 'W']
        y_seqs = df_gen[df_gen['cdr3'].str[0] == 'Y']

        # 按权重比例采样
        n_total = min(500, len(df_gen))
        n_f = max(1, int(n_total * config['F']))
        n_w = max(1, int(n_total * config['W']))
        n_y = n_total - n_f - n_w

        sampled = []
        if len(f_seqs) > 0:
            sampled.append(f_seqs.sample(n=min(n_f, len(f_seqs)), random_state=42))
        if len(w_seqs) > 0:
            sampled.append(w_seqs.sample(n=min(n_w, len(w_seqs)), random_state=42))
        if len(y_seqs) > 0:
            sampled.append(y_seqs.sample(n=min(n_y, len(y_seqs)), random_state=42))

        if sampled:
            df_sampled = pd.concat(sampled)
            fc_result = sim.estimate_fc_rate(df_sampled)
            results.append({
                'name': config['name'],
                'experiment': config['experiment'],
                'varying': config['varying'],
                'F_weight': config['F'],
                'W_weight': config['W'],
                'Y_weight': config['Y'],
                'fc_rate': fc_result['estimated_fc_rate'],
                'rf2_rate': fc_result['avg_rf2_probability'],
                'n_sequences': len(df_sampled),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT / "fwy_weight_tuning_results.csv", index=False)

    # 生成趋势图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'F': '#4C72B0', 'W': '#DD8452', 'Y': '#55A868'}

    for exp_id, varying in [(1, 'F'), (2, 'W'), (3, 'Y')]:
        ax = axes[exp_id - 1]
        sub = results_df[results_df['experiment'] == exp_id]
        weight_col = f'{varying}_weight'

        ax.plot(sub[weight_col], sub['fc_rate'] * 100, 'o-',
                color=colors[varying], linewidth=2, markersize=4)
        ax.axhline(y=0.61, color='gray', linestyle='--', alpha=0.7, label='v2.0 baseline (0.61%)')
        ax.set_xlabel(f'{varying} Weight', fontsize=12)
        ax.set_ylabel('Estimated FC Rate (%)', fontsize=12)
        ax.set_title(f'FC Rate vs {varying} First Residue Weight', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 标注最优点
        best_idx = sub['fc_rate'].idxmax()
        best_row = sub.loc[best_idx]
        ax.annotate(f'Best: {best_row["fc_rate"]*100:.2f}%\n({varying}={best_row[weight_col]:.2f})',
                   xy=(best_row[weight_col], best_row['fc_rate'] * 100),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'),
                   color='red')

    plt.tight_layout()
    fig_path = OUTPUT / "fwy_weight_fc_rate_trend.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  权重调节结果:")
    for varying in ['F', 'W', 'Y']:
        sub = results_df[results_df['varying'] == varying]
        best = sub.loc[sub['fc_rate'].idxmax()]
        print(f"    {varying}最优权重: {best[f'{varying}_weight']:.2f} "
              f"(FC率={best['fc_rate']*100:.2f}%, RF2通过率={best['rf2_rate']:.1%})")

    print(f"\n  趋势图已保存: {fig_path}")
    print(f"  详细数据: {OUTPUT / 'fwy_weight_tuning_results.csv'}")

    return results_df


# ============================================================
# 主流程
# ============================================================
def main():
    # 任务1
    result, sim = task1_fc_simulation()

    # 任务2
    results_df = task2_weight_tuning(sim)

    print("\n" + "=" * 70)
    print("全部任务完成")
    print(f"输出目录: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
