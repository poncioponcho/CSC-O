#!/usr/bin/env python3
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from csco_config import AMINO_ACIDS, AROMATIC, POSITIVE, HYDROPHOBIC, extract_cdr3_features


def load_strategy(strategy_path):
    with open(strategy_path) as f:
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
    if 'hydrophobic_min_ratio' in effective_sp and features['hydrophobic_ratio'] >= effective_sp['hydrophobic_min_ratio']:
        score += 1.0
    return score


def generate_cdr3(strategy, length, n_samples, rng):
    hc = strategy['hard_constraints']
    sp = strategy.get('soft_preferences', {})
    lsp = strategy.get('length_specific_preferences', {})
    len_prefs = lsp.get(str(length), {})
    effective_sp = {**sp, **len_prefs}

    first_whitelist = hc.get('cdr3_first_residue_whitelist', AMINO_ACIDS)
    last_whitelist = hc.get('cdr3_last_residue_whitelist', AMINO_ACIDS)
    preferred_lengths = hc.get('cdr3_length_preferred', [length])
    templates = strategy.get('success_templates', {})
    len_key = f'length_{length}'
    template_list = templates.get(len_key, [])

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
                            if aa in HYDROPHOBIC and 'hydrophobic_min_ratio' in effective_sp:
                                hyd_target = effective_sp['hydrophobic_min_ratio']
                                w *= (1.0 + hyd_target)
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

        has_anti, pattern = check_anti_patterns(cdr3, strategy)
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


def generate_vh_framework(cdr3, framework_templates, rng):
    if not framework_templates:
        return f"EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR{cdr3}WGQGTLVTVSS"
    return rng.choice(framework_templates).replace('{CDR3}', cdr3)


def main():
    parser = argparse.ArgumentParser(description="CSC-O Constrained Sequence Generator")
    parser.add_argument("--strategy", required=True, help="design_strategy.json path")
    parser.add_argument("--output", default="./output/generated_sequences.csv", help="output CSV path")
    parser.add_argument("--n-samples", type=int, default=10000, help="number of sequences to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-soft-score", type=float, default=2.0, help="minimum soft preference score")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    strategy = load_strategy(args.strategy)
    hc = strategy['hard_constraints']
    allowed_lengths = hc.get('cdr3_length_allowed', [5, 6, 7])
    preferred_lengths = hc.get('cdr3_length_preferred', [6, 7])
    length_weights_raw = strategy.get('length_generation_weights', {})

    print(f"CSC-O 序列生成器")
    print(f"  策略: {strategy.get('strategy_name', 'CSC-O_v1')}")
    print(f"  允许长度: {allowed_lengths}")
    print(f"  目标生成数: {args.n_samples}")

    all_generated = []
    if length_weights_raw:
        weight_sum = sum(length_weights_raw.get(str(l), 0.0) for l in allowed_lengths)
        if weight_sum > 0:
            for length in allowed_lengths:
                w = length_weights_raw.get(str(length), 0.0)
                n_target = max(int(args.n_samples * w / weight_sum), 100)
                seqs = generate_cdr3(strategy, length, n_target, rng)
                all_generated.extend(seqs)
                print(f"  长度 {length}: 生成 {len(seqs)} 条 (权重={w:.3f}, 目标 {n_target})")
        else:
            length_weights_raw = {}
    if not length_weights_raw:
        n_per_length = args.n_samples // len(allowed_lengths)
        for length in allowed_lengths:
            n_target = n_per_length * 2 if length in preferred_lengths else n_per_length
            seqs = generate_cdr3(strategy, length, n_target, rng)
            all_generated.extend(seqs)
            print(f"  长度 {length}: 生成 {len(seqs)} 条 (目标 {n_target})")

    if args.min_soft_score > 0:
        before = len(all_generated)
        all_generated = [s for s in all_generated if s['soft_score'] >= args.min_soft_score]
        print(f"  软偏好过滤: {before} → {len(all_generated)} (score >= {args.min_soft_score})")

    from sklearn.metrics.pairwise import cosine_similarity
    if len(all_generated) > 100:
        feat_matrix = np.array([[s['aromatic_ratio'], s['glycine_ratio'], s['serine_ratio'],
                                  s['proline_count'], s['positive_count']] for s in all_generated])
        sim = cosine_similarity(feat_matrix)
        keep = [True] * len(all_generated)
        for i in range(len(all_generated)):
            if not keep[i]:
                continue
            for j in range(i+1, len(all_generated)):
                if not keep[j]:
                    continue
                if sim[i, j] > 0.99 and all_generated[i]['cdr3'] == all_generated[j]['cdr3']:
                    keep[j] = False
        all_generated = [s for s, k in zip(all_generated, keep) if k]
        print(f"  去重后: {len(all_generated)}")

    all_generated.sort(key=lambda x: -x['soft_score'])

    import pandas as pd
    df = pd.DataFrame(all_generated)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  输出: {out_path} ({len(df)} 条)")
    print(f"  CDR3长度分布: {dict(Counter(df['cdr3_len']))}")
    print(f"  平均软偏好得分: {df['soft_score'].mean():.2f}")


if __name__ == "__main__":
    main()
