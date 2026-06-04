#!/usr/bin/env python3
"""
CSC-O v2.4 白名单逻辑验证与FC率提升策略测试
================================================
任务1: 构造F/W/Y首残基长度6测试序列集
任务2: 模拟v2.3白名单逻辑验证过滤效果
任务3: 设计新版v2.4白名单逻辑(长度6-7+首残基F/W/Y+甘氨酸限制等)
任务4: 运行新版脚本生成测试序列集
任务5: 增强日志功能(首残基/长度/过滤结果/原因)
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from csco_config import AMINO_ACIDS, AROMATIC, POSITIVE, HYDROPHOBIC, GLYCINE, SERINE, PROLINE, extract_cdr3_features
from csco_generator import (load_strategy, check_hard_constraints,
                             check_anti_patterns, score_soft_preferences,
                             generate_cdr3, _reject_detail)

OUTPUT_DIR = Path(__file__).parent / "output_v2.4_test"
V23_STRATEGY = Path(__file__).parent / "output_server_v2.3" / "design_strategy.json"


# ============================================================
# 任务1: 构造F/W/Y首残基长度6测试序列集
# ============================================================
def task1_construct_test_sequences():
    print("=" * 70)
    print("任务1: 构造F/W/Y首残基长度6测试序列集")
    print("=" * 70)

    test_sequences = []

    # 类别1: F/W/Y开头 + 合法尾残基 + 各种中间组合
    for first in ['F', 'W', 'Y']:
        # 典型成功模板变体
        for last in ['Y', 'A', 'H', 'S', 'T', 'N', 'D', 'V', 'W', 'F', 'I']:
            # 中间4位覆盖不同氨基酸类别
            middles = [
                'KDSP',   # 常见组合
                'RRRR',   # 高正电荷
                'WWWW',   # 高芳香族
                'GGGG',   # 高甘氨酸(应被反模式或软偏好过滤)
                'SSSS',   # 高丝氨酸(应被反模式过滤)
                'LLLL',   # 含LL反模式
                'AVIL',   # 高疏水性
                'DEKR',   # 混合极性
                'PPPP',   # 高脯氨酸
                'QNMH',   # 极性不带电
                'CWYM',   # 含稀有氨基酸
                'KDGR',   # 含1个甘氨酸
            ]
            for mid in middles:
                seq = first + mid + last
                if len(seq) == 6:
                    test_sequences.append(seq)

    # 类别2: 非白名单首残基(应被过滤)
    for first in ['A', 'G', 'D', 'N', 'I', 'V', 'S', 'L', 'C', 'E', 'H', 'K', 'M', 'P', 'Q', 'R', 'T']:
        test_sequences.append(first + 'KDSP' + 'Y')

    # 类别3: 非白名单尾残基(应被过滤)
    for last in ['C', 'E', 'G', 'K', 'L', 'M', 'P', 'Q', 'R']:
        test_sequences.append('F' + 'KDSP' + last)

    # 类别4: 长度边界
    test_sequences.extend([
        'FDY',       # 长度3
        'FWY',       # 长度3
        'FKDSP',     # 长度5
        'FKDSPYH',   # 长度7
        'FARTGQFTY', # 长度9
    ])

    # 去重
    test_sequences = list(dict.fromkeys(test_sequences))
    print(f"\n  构造测试序列总数: {len(test_sequences)}")
    print(f"  F开头: {sum(1 for s in test_sequences if s[0]=='F')}")
    print(f"  W开头: {sum(1 for s in test_sequences if s[0]=='W')}")
    print(f"  Y开头: {sum(1 for s in test_sequences if s[0]=='Y')}")
    print(f"  其他开头: {sum(1 for s in test_sequences if s[0] not in 'FWY')}")
    return test_sequences


# ============================================================
# 任务2: 模拟v2.3白名单逻辑验证过滤效果
# ============================================================
def task2_simulate_v23_filter(test_sequences):
    print("\n" + "=" * 70)
    print("任务2: 模拟v2.3白名单逻辑验证过滤效果")
    print("=" * 70)

    strategy = load_strategy(str(V23_STRATEGY))
    hc = strategy['hard_constraints']

    print(f"\n  v2.3策略配置:")
    print(f"    允许长度: {hc.get('cdr3_length_allowed', [])}")
    print(f"    首残基白名单: {hc.get('cdr3_first_residue_whitelist', [])}")
    print(f"    尾残基白名单: {hc.get('cdr3_last_residue_whitelist', [])}")
    print(f"    首残基芳香族要求: {hc.get('cdr3_min_aromatic_first', False)}")
    print(f"    反模式: {strategy.get('anti_patterns', [])}")

    passed = []
    filtered = []
    filter_reasons = Counter()

    for seq in test_sequences:
        ok, reason = check_hard_constraints(seq, strategy, verbose=True)
        has_anti, pattern = check_anti_patterns(seq, strategy)
        soft_score = score_soft_preferences(seq, strategy) if ok and not has_anti else 0

        actual_pass = ok and not has_anti
        record = {
            'cdr3': seq,
            'first_residue': seq[0],
            'last_residue': seq[-1],
            'length': len(seq),
            'passed': actual_pass,
            'hard_constraint_ok': ok,
            'reject_reason': reason if not ok else (f'anti_pattern_{pattern}' if has_anti else None),
            'anti_pattern': pattern if has_anti else None,
            'soft_score': soft_score,
        }

        if actual_pass:
            passed.append(record)
        else:
            filtered.append(record)
            r = reason if not ok else f'anti_pattern_{pattern}'
            filter_reasons[r] += 1

    print(f"\n  v2.3过滤结果汇总:")
    print(f"    总测试序列: {len(test_sequences)}")
    print(f"    通过: {len(passed)}")
    print(f"    被过滤: {len(filtered)}")
    print(f"    过滤原因分布:")
    for reason, cnt in filter_reasons.most_common():
        print(f"      {reason}: {cnt}")

    # 分析通过的序列特征
    if passed:
        df_pass = pd.DataFrame(passed)
        print(f"\n  通过序列特征分析:")
        print(f"    首残基分布: {dict(df_pass['first_residue'].value_counts())}")
        print(f"    尾残基分布: {dict(df_pass['last_residue'].value_counts())}")
        print(f"    长度分布: {dict(df_pass['length'].value_counts())}")
        print(f"    平均软偏好得分: {df_pass['soft_score'].mean():.2f}")

    return passed, filtered, strategy


# ============================================================
# 任务3: 设计新版v2.4白名单逻辑
# ============================================================
def task3_design_v24_strategy(v23_strategy):
    print("\n" + "=" * 70)
    print("任务3: 设计新版v2.4白名单逻辑")
    print("=" * 70)

    v24 = json.loads(json.dumps(v23_strategy))  # 深拷贝

    # 修改1: CDR3长度限制为6-7 (基于因果证据: 长度6-7 RF2通过率最高)
    v24['hard_constraints']['cdr3_length_allowed'] = [6, 7]
    v24['hard_constraints']['cdr3_length_preferred'] = [6, 7]

    # 修改2: 首残基强制F/W/Y (保持v2.3)
    v24['hard_constraints']['cdr3_first_residue_whitelist'] = ['F', 'W', 'Y']
    v24['hard_constraints']['cdr3_min_aromatic_first'] = True

    # 修改3: 尾残基白名单收紧至高频FC尾残基
    # 基于v2.3数据分析: Y(17.2%), A(10.6%), H(9.4%), N(8.8%), D(8.2%), W(8.2%)
    v24['hard_constraints']['cdr3_last_residue_whitelist'] = ['Y', 'A', 'H', 'N', 'D', 'W', 'S', 'T', 'V', 'F']

    # 修改4: 甘氨酸限制收紧 (Cox HR=4.35, 最强风险因子)
    v24['soft_preferences']['glycine_max_ratio'] = 0.15  # 从0.20收紧至0.15
    v24['length_specific_preferences']['6']['glycine_max_ratio'] = 0.15
    v24['length_specific_preferences']['7']['glycine_max_ratio'] = 0.15

    # 修改5: 长度6芳香族要求适度提高 (分层ATE=-1.91, 有利)
    v24['length_specific_preferences']['6']['aromatic_min_ratio'] = 0.17  # 从0.15微调

    # 修改6: 删除长度9+的配置(不再生成)
    for key in ['9', '11', '12', '13']:
        v24['length_specific_preferences'].pop(key, None)

    # 修改7: 长度生成权重全部集中在6-7
    v24['length_generation_weights'] = {
        '6': 0.4965,
        '7': 0.7247,
    }

    # 修改8: 删除长度9+的success_templates
    for key in ['length_9', 'length_11', 'length_12', 'length_13']:
        v24['success_templates'].pop(key, None)

    # 修改9: 仅保留首残基为F/W/Y的template
    for len_key in ['length_6', 'length_7']:
        if len_key in v24['success_templates']:
            v24['success_templates'][len_key] = [
                t for t in v24['success_templates'][len_key]
                if t[0] in ['F', 'W', 'Y']
            ]

    # 更新版本信息
    v24['strategy_name'] = 'CSC-O_v2.4'
    v24['version'] = '2.4'
    v24['description'] = 'FC-rate-optimized: length 6-7 only, first=F/W/Y, glycine<=0.15'

    print(f"\n  v2.4策略变更:")
    print(f"    CDR3长度: {v23_strategy['hard_constraints']['cdr3_length_allowed']} → [6, 7]")
    print(f"    首残基白名单: {v24['hard_constraints']['cdr3_first_residue_whitelist']} (不变)")
    print(f"    尾残基白名单: {v23_strategy['hard_constraints']['cdr3_last_residue_whitelist']}")
    print(f"                  → {v24['hard_constraints']['cdr3_last_residue_whitelist']} (移除I)")
    print(f"    glycine_max_ratio: 0.20 → 0.15 (全局+长度6/7)")
    print(f"    长度6 aromatic_min_ratio: 0.15 → 0.17")
    print(f"    长度9+配置: 已移除")
    print(f"    长度9+ templates: 已移除")
    print(f"    非F/W/Y开头的templates: 已过滤")

    # 保存v2.4策略
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_path = OUTPUT_DIR / "design_strategy_v2.4.json"
    with open(strategy_path, 'w', encoding='utf-8') as f:
        json.dump(v24, f, indent=2, ensure_ascii=False)
    print(f"\n  v2.4策略已保存: {strategy_path}")

    return v24


# ============================================================
# 任务4+5: 运行新版脚本生成测试序列 + 增强日志
# ============================================================
def task4_5_generate_with_logging(v24_strategy):
    print("\n" + "=" * 70)
    print("任务4+5: 运行v2.4脚本生成测试序列 + 增强日志")
    print("=" * 70)

    rng = random.Random(42)
    np.random.seed(42)
    hc = v24_strategy['hard_constraints']

    print(f"\n  v2.4生成配置:")
    print(f"    允许长度: {hc['cdr3_length_allowed']}")
    print(f"    首残基白名单: {hc['cdr3_first_residue_whitelist']}")
    print(f"    尾残基白名单: {hc['cdr3_last_residue_whitelist']}")
    print(f"    首残基芳香族要求: {hc['cdr3_min_aromatic_first']}")
    print(f"    反模式: {v24_strategy.get('anti_patterns', [])}")
    print(f"    glycine_max_ratio: {v24_strategy['soft_preferences']['glycine_max_ratio']}")

    all_generated = []
    filter_log = []
    n_samples = 1000

    for length in hc['cdr3_length_allowed']:
        print(f"\n  --- 生成长度={length}的序列 ---")
        seqs = generate_cdr3(v24_strategy, length, n_samples, rng,
                             verbose=True, filter_log=filter_log)
        all_generated.extend(seqs)
        print(f"  长度 {length}: 生成 {len(seqs)} 条")

    # 软偏好过滤
    min_soft = 1.5
    before = len(all_generated)
    all_generated = [s for s in all_generated if s['soft_score'] >= min_soft]
    print(f"\n  软偏好过滤: {before} → {len(all_generated)} (score >= {min_soft})")

    # 去重
    seen = set()
    unique = []
    for s in all_generated:
        if s['cdr3'] not in seen:
            seen.add(s['cdr3'])
            unique.append(s)
    all_generated = unique
    print(f"  去重后: {len(all_generated)}")

    # 排序
    all_generated.sort(key=lambda x: -x['soft_score'])

    # 保存结果
    df = pd.DataFrame(all_generated)
    csv_path = OUTPUT_DIR / "generated_sequences_v2.4.csv"
    df.to_csv(csv_path, index=False)

    # 保存过滤日志
    if filter_log:
        log_df = pd.DataFrame(filter_log)
        log_path = OUTPUT_DIR / "filter_log_v2.4.csv"
        log_df.to_csv(log_path, index=False)

    # ============================================================
    # 增强日志: 逐序列详细报告
    # ============================================================
    print(f"\n" + "=" * 70)
    print("增强日志: 逐序列详细报告 (前30条)")
    print("=" * 70)
    print(f"  {'序列':^8s} | {'首残基':^4s} | {'尾残基':^4s} | {'长度':^3s} | {'软分':^4s} | {'芳香%':^5s} | {'甘氨%':^5s} | {'疏水%':^5s} | {'结果':^6s} | {'原因'}")
    print(f"  {'-'*8} | {'-'*4} | {'-'*4} | {'-'*3} | {'-'*4} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*20}")

    # 显示通过的序列
    for i, s in enumerate(all_generated[:20]):
        print(f"  {s['cdr3']:^8s} | {s['cdr3'][0]:^4s} | {s['cdr3'][-1]:^4s} | {s['cdr3_len']:^3d} | {s['soft_score']:^4.1f} | {s['aromatic_ratio']:^5.2f} | {s['glycine_ratio']:^5.2f} | {s['hydrophobic_ratio']:^5.2f} | {'通过':^6s} | 满足全部约束")

    # 显示被过滤的序列
    if filter_log:
        print(f"\n  被过滤序列示例 (前10条):")
        for i, rec in enumerate(filter_log[:10]):
            first_r = rec.get('first_residue', rec['cdr3'][0] if 'cdr3' in rec else '?')
            last_r = rec.get('last_residue', rec['cdr3'][-1] if 'cdr3' in rec else '?')
            print(f"  {rec.get('cdr3','?'):^8s} | {first_r:^4s} | {last_r:^4s} | {rec.get('length','?'):^3} | {'--':^4} | {'--':^5} | {'--':^5} | {'--':^5} | {'拒绝':^6} | {rec.get('reject_detail', rec.get('reject_reason', '?'))}")

    # ============================================================
    # 统计汇总
    # ============================================================
    print(f"\n" + "=" * 70)
    print("v2.4生成结果统计汇总")
    print("=" * 70)

    print(f"\n  总生成序列: {len(all_generated)}")
    print(f"  总被过滤: {len(filter_log)}")

    if all_generated:
        df_gen = pd.DataFrame(all_generated)
        print(f"\n  首残基分布:")
        for aa in ['F', 'W', 'Y']:
            cnt = (df_gen['cdr3'].str[0] == aa).sum()
            pct = cnt / len(df_gen) * 100
            print(f"    {aa}: {cnt} ({pct:.1f}%)")

        print(f"\n  尾残基分布:")
        last_dist = df_gen['cdr3'].str[-1].value_counts()
        for aa, cnt in last_dist.items():
            print(f"    {aa}: {cnt} ({cnt/len(df_gen)*100:.1f}%)")

        print(f"\n  长度分布:")
        len_dist = df_gen['cdr3_len'].value_counts().sort_index()
        for l, cnt in len_dist.items():
            print(f"    长度{l}: {cnt} ({cnt/len(df_gen)*100:.1f}%)")

        print(f"\n  特征统计:")
        print(f"    平均芳香族比例: {df_gen['aromatic_ratio'].mean():.3f}")
        print(f"    平均甘氨酸比例: {df_gen['glycine_ratio'].mean():.3f}")
        print(f"    平均丝氨酸比例: {df_gen['serine_ratio'].mean():.3f}")
        print(f"    平均疏水比例:   {df_gen['hydrophobic_ratio'].mean():.3f}")
        print(f"    平均软偏好得分: {df_gen['soft_score'].mean():.2f}")

        # 甘氨酸超标检查
        gly_high = (df_gen['glycine_ratio'] > 0.15).sum()
        print(f"    甘氨酸>0.15的序列: {gly_high} ({gly_high/len(df_gen)*100:.1f}%)")

    if filter_log:
        log_df = pd.DataFrame(filter_log)
        print(f"\n  过滤原因分布:")
        for reason, cnt in log_df['reject_reason'].value_counts().items():
            print(f"    {reason}: {cnt}")

    # ============================================================
    # v2.3 vs v2.4 对比
    # ============================================================
    print(f"\n" + "=" * 70)
    print("v2.3 vs v2.4 对比")
    print("=" * 70)

    print(f"\n  {'指标':<25s} | {'v2.3':>12s} | {'v2.4':>12s} | {'变化'}")
    print(f"  {'-'*25} | {'-'*12} | {'-'*12} | {'-'*15}")

    # v2.3数据来自之前的运行
    v23_total = 791
    v23_lengths = {6: 158, 7: 233, 9: 100, 11: 100, 12: 100, 13: 100}
    v24_total = len(all_generated) if all_generated else 0
    v24_lengths = dict(Counter(df_gen['cdr3_len'])) if all_generated else {}

    print(f"  {'总序列数':<25s} | {v23_total:>12d} | {v24_total:>12d} | {'+' if v24_total > v23_total else ''}{v24_total - v23_total}")
    print(f"  {'允许长度范围':<25s} | {'6,7,9,11,12,13':>12s} | {'6,7':>12s} | 收紧至6-7")
    print(f"  {'首残基白名单':<25s} | {'[F,W,Y]':>12s} | {'[F,W,Y]':>12s} | 不变")
    print(f"  {'glycine_max_ratio':<25s} | {'0.20':>12s} | {'0.15':>12s} | 收紧0.05")

    if all_generated:
        v24_gly_mean = df_gen['glycine_ratio'].mean()
        print(f"  {'平均甘氨酸比例':<25s} | {'--':>12s} | {v24_gly_mean:>12.3f} | v2.4新增指标")
        v24_soft_mean = df_gen['soft_score'].mean()
        print(f"  {'平均软偏好得分':<25s} | {'3.99':>12s} | {v24_soft_mean:>12.2f} | {'+' if v24_soft_mean > 3.99 else ''}{v24_soft_mean - 3.99:.2f}")

    print(f"\n  输出文件:")
    print(f"    策略文件: {OUTPUT_DIR / 'design_strategy_v2.4.json'}")
    print(f"    生成序列: {OUTPUT_DIR / 'generated_sequences_v2.4.csv'}")
    if filter_log:
        print(f"    过滤日志: {OUTPUT_DIR / 'filter_log_v2.4.csv'}")

    return all_generated, filter_log


# ============================================================
# 主流程
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 任务1
    test_sequences = task1_construct_test_sequences()

    # 任务2
    passed, filtered, v23_strategy = task2_simulate_v23_filter(test_sequences)

    # 任务3
    v24_strategy = task3_design_v24_strategy(v23_strategy)

    # 任务4+5
    all_generated, filter_log = task4_5_generate_with_logging(v24_strategy)

    print("\n" + "=" * 70)
    print("全部任务完成")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
