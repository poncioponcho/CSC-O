#!/usr/bin/env python3
"""
验证脚本：模拟移除 Length-5 后的序列分布变化
对比 baseline（含len5）vs no_len5（移除+权重重分配）两种策略的生成效果
"""
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from csco_config import AMINO_ACIDS, AROMATIC, extract_cdr3_features


def load_strategy(path):
    with open(path) as f:
        return json.load(f)


def check_hard_constraints(cdr3, strategy, length_specific_prefs=None):
    hc = strategy['hard_constraints']
    features = extract_cdr3_features(cdr3)
    if features['cdr3_len'] not in hc.get('cdr3_length_allowed', [5, 6, 7]):
        return False, "length_not_allowed"
    lsp = length_specific_prefs or strategy.get('length_specific_preferences', {}).get(str(features['cdr3_len']), {})
    if hc.get('cdr3_min_aromatic_first') and not lsp.get('first_aromatic_optional'):
        if not features['first_is_aromatic']:
            return False, "first_not_aromatic"
    if hc.get('cdr3_first_residue_whitelist') and cdr3[0] not in hc['cdr3_first_residue_whitelist']:
        return False, "first_not_whitelisted"
    if hc.get('cdr3_last_residue_whitelist') and cdr3[-1] not in hc['cdr3_last_residue_whitelist']:
        return False, "last_not_whitelisted"
    if features['positive_count'] < hc.get('cdr3_min_positive_count', 0):
        return False, "insufficient_positive"
    return True, None


def check_anti_patterns(cdr3, strategy):
    for pattern in strategy.get('anti_patterns', []):
        if pattern in cdr3:
            return True, pattern
    return False, None


def score_soft_preferences(cdr3, strategy):
    sp = strategy.get('soft_preferences', {})
    lsp_all = strategy.get('length_specific_preferences', {})
    features = extract_cdr3_features(cdr3)
    len_key = str(features['cdr3_len'])
    effective_sp = {**sp, **lsp_all.get(len_key, {})}
    score = 0.0
    if 'aromatic_min_ratio' in effective_sp and features['aromatic_ratio'] >= effective_sp['aromatic_min_ratio']:
        score += 1.0
    if 'glycine_max_ratio' in effective_sp and features['glycine_ratio'] <= effective_sp['glycine_max_ratio']:
        score += 1.0
    if 'serine_max_ratio' in effective_sp and features['serine_ratio'] <= effective_sp.get('serine_max_ratio', 0.15):
        score += 1.0
    if 'proline_max_count' in effective_sp and features['proline_count'] <= effective_sp.get('proline_max_count', 1):
        score += 0.5
    return score


def generate_cdr3(strategy, length, n_samples, rng):
    hc = strategy['hard_constraints']
    sp = strategy.get('soft_preferences', {})
    lsp = strategy.get('length_specific_preferences', {})
    len_prefs = lsp.get(str(length), {})
    effective_sp = {**sp, **len_prefs}

    first_whitelist = hc.get('cdr3_first_residue_whitelist', AMINO_ACIDS)
    last_whitelist = hc.get('cdr3_last_residue_whitelist', AMINO_ACIDS)
    templates = strategy.get('success_templates', {})
    template_list = templates.get(f'length_{length}', [])

    generated = []
    attempts = 0
    max_attempts = n_samples * 50

    while len(generated) < n_samples and attempts < max_attempts:
        attempts += 1
        if template_list and rng.random() < 0.3:
            template = rng.choice(template_list)
            pos = rng.randint(0, len(template) - 1)
            candidates = [a for a in AMINO_ACIDS if a != template[pos]]
            cdr3 = template[:pos] + rng.choice(candidates) + template[pos+1:]
        else:
            first = rng.choice(first_whitelist)
            last = rng.choice(last_whitelist)
            middle_len = length - 2
            if middle_len > 0:
                middle = []
                for _ in range(middle_len):
                    if effective_sp.get('aromatic_min_ratio', 0) > 0 and rng.random() < effective_sp['aromatic_min_ratio']:
                        middle.append(rng.choice(list(AROMATIC)))
                    else:
                        weights = []
                        for aa in AMINO_ACIDS:
                            w = 1.0
                            if aa == 'G' and 'glycine_max_ratio' in effective_sp:
                                w *= max(0.1, 1.0 - effective_sp['glycine_max_ratio'])
                            if aa == 'S' and 'serine_max_ratio' in effective_sp:
                                w *= max(0.1, 1.0 - effective_sp['serine_max_ratio'])
                            if aa == 'P':
                                pro_max = effective_sp.get('proline_max_count', 1)
                                w *= 0.3 if pro_max <= 1 else 0.6
                            weights.append(w)
                        total = sum(weights)
                        weights = [w/total for w in weights]
                        middle.append(rng.choices(AMINO_ACIDS, weights=weights, k=1)[0])
                cdr3 = first + ''.join(middle) + last
            else:
                cdr3 = first + last

        if len(cdr3) != length:
            continue

        ok, reason = check_hard_constraints(cdr3, strategy)
        if not ok:
            continue
        has_anti, _ = check_anti_patterns(cdr3, strategy)
        if has_anti:
            continue
        soft_score = score_soft_preferences(cdr3, strategy)
        features = extract_cdr3_features(cdr3)
        generated.append({
            'cdr3': cdr3,
            'soft_score': soft_score,
            **{k: v for k, v in features.items() if isinstance(v, (int, float, bool))},
        })
    return generated


def run_generation(strategy, label, n_total=8816, seed=42):
    rng = random.Random(seed)
    np.random.seed(seed)

    allowed_lengths = strategy['hard_constraints'].get('cdr3_length_allowed', [5, 6, 7])
    length_weights_raw = strategy.get('length_generation_weights', {})

    all_generated = []
    gen_stats = {}

    if length_weights_raw:
        weight_sum = sum(length_weights_raw.get(str(l), 0.0) for l in allowed_lengths)
        if weight_sum > 0:
            for length in allowed_lengths:
                w = length_weights_raw.get(str(length), 0.0)
                n_target = max(int(n_total * w / weight_sum), 100)
                seqs = generate_cdr3(strategy, length, n_target, rng)
                gen_stats[length] = {'target': n_target, 'generated': len(seqs),
                                     'weight': round(w, 4), 'pct': round(w/weight_sum*100, 1)}
                all_generated.extend(seqs)
        else:
            length_weights_raw = {}
    if not length_weights_raw:
        n_per = n_total // len(allowed_lengths)
        for length in allowed_lengths:
            n_target = n_per * 2 if length in [6, 7] else n_per
            seqs = generate_cdr3(strategy, length, n_target, rng)
            gen_stats[length] = {'target': n_target, 'generated': len(seqs)}
            all_generated.extend(seqs)

    before_filter = len(all_generated)
    all_generated = [s for s in all_generated if s['soft_score'] >= 2.0]
    after_filter = len(all_generated)

    df = pd.DataFrame(all_generated)
    dist = dict(Counter(df['cdr3_len'])) if len(df) > 0 else {}

    print(f"\n{'='*60}")
    print(f"策略: {label}")
    print(f"{'='*60}")
    print(f"  允许长度: {allowed_lengths}")
    print(f"  权重配置: {length_weights_raw}")

    print(f"\n  各长度生成统计:")
    print(f"  {'Length':>6s} {'权重':>8s} {'占比%':>7s} {'目标数':>7s} {'实际生成':>8s} {'达成率':>7s}")
    print(f"  {'-'*50}")
    for l in sorted(gen_stats.keys()):
        s = gen_stats[l]
        rate = s['generated'] / s['target'] * 100 if s['target'] > 0 else 0
        wt = s.get('weight', '-')
        pct = s.get('pct', '-')
        print(f"  {l:>6d} {str(wt):>8s} {str(pct):>7s} {s['target']:>7d} {s['generated']:>8d} {rate:>6.1f}%")

    print(f"\n  软偏好过滤 (score>=2.0): {before_filter} → {after_filter}")
    print(f"  最终序列分布: {dist}")
    print(f"  总计: {after_filter} 条")

    if len(df) > 0:
        print(f"\n  特征均值:")
        for col in ['aromatic_ratio', 'glycine_ratio', 'serine_ratio',
                     'hydrophobic_ratio', 'proline_count', 'positive_count']:
            if col in df.columns:
                print(f"    {col:<20s}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    return df, gen_stats, dist


def compare_distributions(baseline_dist, no_len5_dist, baseline_df, no_len5_df):
    print(f"\n{'='*60}")
    print(f"对比分析: Baseline vs No-Length-5")
    print(f"{'='*60}")

    all_lens = sorted(set(list(baseline_dist.keys()) + list(no_len5_dist.keys())))
    total_base = sum(baseline_dist.values())
    total_no5 = sum(no_len5_dist.values())

    print(f"\n  {'Length':>6s} {'Baseline':>10s} {'Base%':>7s} {'No-Len5':>10s} {'No5%':>7s} {'变化':>10s}")
    print(f"  {'-'*55}")
    for l in all_lens:
        b = baseline_dist.get(l, 0)
        n = no_len5_dist.get(l, 0)
        bp = b / total_base * 100 if total_base > 0 else 0
        np_ = n / total_no5 * 100 if total_no5 > 0 else 0
        delta = np_ - bp
        arrow = "↑" if delta > 1 else "↓" if delta < -1 else "="
        print(f"  {l:>6d} {b:>10d} {bp:>6.1f}% {n:>10d} {np_:>6.1f}% {delta:>+9.1f}% {arrow}")

    print(f"\n  总计: Baseline={total_base}, No-Len5={total_no5}, 变化={total_no5-total_base:+d}")

    if len(baseline_df) > 0 and len(no_len5_df) > 0:
        print(f"\n  特征均值变化:")
        print(f"  {'特征':<20s} {'Baseline':>10s} {'No-Len5':>10s} {'Delta':>10s}")
        print(f"  {'-'*55}")
        for col in ['aromatic_ratio', 'glycine_ratio', 'serine_ratio',
                     'hydrophobic_ratio', 'proline_count', 'positive_count']:
            if col in baseline_df.columns and col in no_len5_df.columns:
                bv = baseline_df[col].mean()
                nv = no_len5_df[col].mean()
                d = nv - bv
                arrow = "↑" if d > 0.005 else "↓" if d < -0.005 else "="
                print(f"  {col:<20s} {bv:>10.4f} {nv:>10.4f} {d:>+10.4f} {arrow}")


def main():
    data_dir = Path("/home/test/CSC-O/output_v5")
    strategy_path = data_dir / "design_strategy.json"

    if not strategy_path.exists():
        alt = Path("/home/test/CSC-O/output") / "design_strategy.json"
        if alt.exists():
            strategy_path = alt
        else:
            print("ERROR: design_strategy.json not found")
            return

    base_strategy = load_strategy(strategy_path)

    print("=" * 60)
    print("Length-5 移除验证实验")
    print("=" * 60)
    print(f"  策略文件: {strategy_path}")
    print(f"  原始允许长度: {base_strategy['hard_constraints'].get('cdr3_length_allowed')}")
    print(f"  原始权重: {base_strategy.get('length_generation_weights', {})}")

    print("\n>>> [实验A] Baseline: 当前策略（含Length-5）")
    baseline_df, base_stats, base_dist = run_generation(base_strategy, "Baseline (v5)")

    no_len5_strategy = json.loads(json.dumps(base_strategy))

    old_allowed = list(no_len5_strategy['hard_constraints']['cdr3_length_allowed'])
    old_weights = dict(no_len5_strategy.get('length_generation_weights', {}))
    new_allowed = [l for l in old_allowed if l != 5]
    no_len5_strategy['hard_constraints']['cdr3_length_allowed'] = new_allowed

    if old_weights:
        remaining_weight = sum(old_weights.get(str(l), 0) for l in new_allowed)
        new_weights = {}
        for l in new_allowed:
            raw_w = old_weights.get(str(l), 0)
            new_weights[str(l)] = round(raw_w / remaining_weight, 4) if remaining_weight > 0 else round(1.0/len(new_allowed), 4)
        no_len5_strategy['length_generation_weights'] = new_weights

    print("\n>>> [实验B] No-Len5: 移除Length-5 + 权重重分配")
    print(f"  新允许长度: {new_allowed}")
    print(f"  新权重: {new_weights}")

    no_len5_df, no5_stats, no5_dist = run_generation(no_len5_strategy, "No-Length-5")

    compare_distributions(base_dist, no5_dist, baseline_df, no_len5_df)

    print(f"\n{'='*60}")
    print("结论与建议")
    print(f"{'='*60}")
    total_base = sum(base_dist.values())
    total_no5 = sum(no5_dist.values())
    len5_waste = base_dist.get(5, 0)
    len5_pct = len5_waste / total_base * 100 if total_base > 0 else 0
    len7_gain = no5_dist.get(7, 0) - base_dist.get(7, 0)

    print(f"  1. Length-5 浪费量: {len5_waste} 条 ({len5_pct:.1f}% of total)")
    print(f"  2. 移除后总生成量: {total_base} → {total_no5} ({total_no5-total_base:+d})")
    print(f"  3. Length-7 增量: {base_dist.get(7,0)} → {no5_dist.get(7,0)} ({len7_gain:+d})")
    print(f"  4. Length-6 占比变化: "
          f"{base_dist.get(6,0)/total_base*100:.1f}% → {no5_dist.get(6,0)/total_no5*100:.1f}%")

    out_path = data_dir / "len5_removal_validation.json"
    result = {
        'baseline': {'distribution': base_dist, 'total': total_base, 'stats': base_stats},
        'no_len5': {'distribution': no5_dist, 'total': total_no5, 'stats': no5_stats},
        'summary': {
            'len5_waste': len5_waste,
            'len5_waste_pct': round(len5_pct, 2),
            'total_delta': total_no5 - total_base,
            'len7_delta': len7_gain,
            'new_weights': new_weights if old_weights else None,
        }
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
