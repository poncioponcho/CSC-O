#!/usr/bin/env python3
"""
CSC-O Phase 4: 漏斗感知序列生成器 (Funnel-Aware Sequence Generator)

核心功能:
    基于P(final)而非P(RF2)生成CDR3序列。使用Monte Carlo采样生成候选序列，
    用漏斗感知策略评分，按P(final)排序并应用多样性过滤。

与v2.4生成器的关键区别:
    1. 优化目标: P(final) = P(RF2) × P(final|RF2), 而非仅P(RF2)
    2. 首残基约束: 软偏好(含G条件性), 而非硬约束[仅F/W/Y]
    3. 阶段特异性约束: 从HR_stage1/2提取, 区分RF2和Final效应
    4. 多样性过滤: 基于编辑距离, 确保生成序列集的多样性

设计原则:
    - 兼容现有csco_generator.py的接口
    - 可配置的阶段权重和约束
    - 高效的批量生成和评分

依赖:
    - csco_funnel_aware_strategy.py: FunnelAwareStrategy
    - csco_funnel_counterfactual.py: FunnelAwareCounterfactual (可选, 用于优化)
    - csco_generator.py: 基础生成函数 (兼容)
    - csco_config.py: extract_cdr3_features, AMINO_ACIDS

作者: CSC-O Team
版本: v3.1
日期: 2026-06-04
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import time

from csco_config import (
    AMINO_ACIDS, AROMATIC, GLYCINE, SERINE, PROLINE, HYDROPHOBIC,
    POSITIVE, NEGATIVE,
    extract_cdr3_features,
)
from csco_funnel_aware_strategy import FunnelAwareStrategy, DEFAULT_FUNNEL_WEIGHTS
from csco_generator import (
    load_strategy, check_hard_constraints, check_anti_patterns,
    score_soft_preferences, generate_cdr3,
)


# ═══════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class GeneratedSequence:
    """
    生成序列的完整信息
    
    Attributes:
        cdr3: CDR3序列
        length: 序列长度
        first_aa: 首残基
        rf2_score: RF2阶段评分
        final_score: Final阶段评分
        combined_score: 组合评分
        estimated_p_rf2: 估计P(RF2)
        estimated_p_final_given_rf2: 估计P(final|RF2)
        estimated_p_final: 估计P(final)
        soft_score: v2.4兼容软偏好得分
        diversity_score: 多样性评分 (与已选序列的平均编辑距离)
        recommendation: 改进建议
    """
    cdr3: str
    length: int = 0
    first_aa: str = ""
    rf2_score: float = 0.0
    final_score: float = 0.0
    combined_score: float = 0.0
    estimated_p_rf2: float = 0.0
    estimated_p_final_given_rf2: float = 0.0
    estimated_p_final: float = 0.0
    soft_score: float = 0.0
    diversity_score: float = 0.0
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sequence': self.cdr3,
            'length': self.length,
            'first_aa': self.first_aa,
            'predicted_p_rf2': round(self.estimated_p_rf2, 6),
            'predicted_p_final_given_rf2': round(self.estimated_p_final_given_rf2, 6),
            'predicted_p_final': round(self.estimated_p_final, 6),
            'rf2_score': round(self.rf2_score, 4),
            'final_score': round(self.final_score, 4),
            'combined_score': round(self.combined_score, 4),
            'soft_score': round(self.soft_score, 2),
            'diversity_score': round(self.diversity_score, 2),
            'recommended_mutation': self.recommendation,
        }


# ═══════════════════════════════════════════════════════════════
# 核心类: FunnelAwareGenerator
# ═══════════════════════════════════════════════════════════════

class FunnelAwareGenerator:
    """
    漏斗感知序列生成器
    
    基于P(final)生成CDR3序列, 使用Monte Carlo采样 + 漏斗感知评分 +
    多样性过滤。
    
    使用示例:
        >>> strategy = FunnelAwareStrategy()
        >>> gen = FunnelAwareGenerator(strategy)
        >>> sequences = gen.generate(n_samples=1000, top_n=500)
        >>> print(f"生成 {len(sequences)} 条序列")
        >>> print(f"平均P(final): {np.mean([s.estimated_p_final for s in sequences]):.6f}")
    
    Args:
        strategy: FunnelAwareStrategy实例
        base_strategy_path: v3.x策略JSON文件路径 (用于硬约束)
        funnel_weights: 阶段权重字典, 如 {'rf2': 0.4, 'final': 0.6}
        min_edit_distance: 多样性过滤的最小编辑距离
        verbose: 是否输出详细日志
    """

    def __init__(
        self,
        strategy: FunnelAwareStrategy = None,
        base_strategy_path: str = None,
        funnel_weights: Dict[str, float] = None,
        min_edit_distance: int = 3,
        verbose: bool = True,
    ):
        # 参数验证
        if min_edit_distance < 0 or min_edit_distance > 10:
            raise ValueError(f"min_edit_distance must be 0-10, got {min_edit_distance}")

        self.strategy = strategy or FunnelAwareStrategy(verbose=False)
        self.funnel_weights = funnel_weights or DEFAULT_FUNNEL_WEIGHTS
        self.min_edit_distance = min_edit_distance
        self.verbose = verbose

        # 加载基础策略(用于硬约束)
        self.base_strategy = None
        if base_strategy_path:
            self.base_strategy = load_strategy(base_strategy_path)
        else:
            # 使用v3.1默认策略
            default_path = Path(__file__).parent / 'output_v3_funnel' / 'design_strategy_v3.0.json'
            if default_path.exists():
                self.base_strategy = load_strategy(str(default_path))

        if self.verbose:
            print(f"[FunnelAwareGenerator] 初始化完成")
            print(f"  阶段权重: {self.funnel_weights}")
            print(f"  最小编辑距离: {self.min_edit_distance}")
            print(f"  基础策略: {'已加载' if self.base_strategy else '未加载(将使用默认)'}")

    def generate(
        self,
        n_samples: int = 10000,
        top_n: int = 500,
        allowed_lengths: List[int] = None,
        min_combined_score: float = 0.0,
        seed: int = 42,
    ) -> List[GeneratedSequence]:
        """
        生成漏斗感知优化的CDR3序列
        
        流程:
        1. Monte Carlo采样候选序列 (使用基础策略的硬约束)
        2. 用漏斗感知策略评分每个候选
        3. 估计P(RF2), P(final|RF2), P(final)
        4. 按P(final)排序
        5. 多样性过滤 (编辑距离 > min_edit_distance)
        6. 返回top_n序列
        
        Args:
            n_samples: 采样候选数, 需为正整数
            top_n: 返回的最大序列数, 需为正整数
            allowed_lengths: 允许的CDR3长度列表
            min_combined_score: 最小组合评分阈值
            seed: 随机种子
            
        Returns:
            GeneratedSequence列表, 按estimated_p_final降序
            
        Raises:
            ValueError: 参数无效
        """
        # 输入验证
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError(f"top_n must be a positive integer, got {top_n}")
        if min_combined_score < 0:
            raise ValueError(f"min_combined_score must be non-negative, got {min_combined_score}")

        t0 = time.time()
        rng = random.Random(seed)
        np.random.seed(seed)

        if allowed_lengths is None:
            allowed_lengths = [6, 7]

        if self.verbose:
            print(f"\n[FunnelAwareGenerator] 开始生成")
            print(f"  采样数: {n_samples}, Top-N: {top_n}")
            print(f"  允许长度: {allowed_lengths}")

        # === Step 1: Monte Carlo采样 ===
        candidates = self._sample_candidates(n_samples, allowed_lengths, rng)

        if self.verbose:
            print(f"  采样完成: {len(candidates)} 条候选")

        # === Step 2: 漏斗感知评分 ===
        scored = self._score_candidates(candidates)

        if self.verbose:
            print(f"  评分完成: {len(scored)} 条有效候选")

        # === Step 3: 过滤 ===
        if min_combined_score > 0:
            scored = [s for s in scored if s.combined_score >= min_combined_score]
            if self.verbose:
                print(f"  评分过滤: {len(scored)} 条通过 (combined_score >= {min_combined_score})")

        # === Step 4: 排序 ===
        scored.sort(key=lambda s: -s.estimated_p_final)

        # === Step 5: 多样性过滤 ===
        selected = self._diversity_filter(scored, top_n)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"  多样性过滤: {len(selected)} 条最终序列")
            if selected:
                avg_p_final = np.mean([s.estimated_p_final for s in selected])
                print(f"  平均P(final): {avg_p_final:.6f}")
                print(f"  耗时: {elapsed:.1f}s")
                # 详细日志: 每20条打印P(final)和多样性
                for i, s in enumerate(selected):
                    if (i + 1) % 20 == 0:
                        batch = selected[:i + 1]
                        avg_pf = np.mean([x.estimated_p_final for x in batch])
                        avg_div = np.mean([x.diversity_score for x in batch])
                        print(f"  Step {i+1:3d}: P(final)_avg={avg_pf:.6f}, diversity_avg={avg_div:.2f}")

        return selected

    def _sample_candidates(
        self,
        n_samples: int,
        allowed_lengths: List[int],
        rng: random.Random,
    ) -> List[str]:
        """
        Monte Carlo采样候选序列
        
        使用基础策略的硬约束进行采样, 确保生成的序列满足基本要求。
        如果没有基础策略, 使用简化的随机采样。
        """
        candidates = []
        seen = set()

        if self.base_strategy:
            # 使用现有生成器
            for length in allowed_lengths:
                n_per = n_samples // len(allowed_lengths)
                seqs = generate_cdr3(self.base_strategy, length, n_per, rng, verbose=False)
                for s in seqs:
                    cdr3 = s['cdr3']
                    if cdr3 not in seen:
                        seen.add(cdr3)
                        candidates.append(cdr3)
        else:
            # 简化随机采样
            first_whitelist = ['F', 'W', 'Y', 'V', 'A', 'D', 'T', 'G']
            last_whitelist = ['Y', 'A', 'H', 'N', 'D', 'W', 'S', 'T', 'V', 'F']

            attempts = 0
            max_attempts = n_samples * 10

            while len(candidates) < n_samples and attempts < max_attempts:
                attempts += 1
                length = rng.choice(allowed_lengths)
                first = rng.choice(first_whitelist)
                last = rng.choice(last_whitelist)
                middle_len = length - 2

                if middle_len > 0:
                    middle = []
                    for _ in range(middle_len):
                        middle.append(rng.choice(list(AMINO_ACIDS)))
                    cdr3 = first + ''.join(middle) + last
                else:
                    cdr3 = first + last

                if cdr3 not in seen:
                    seen.add(cdr3)
                    candidates.append(cdr3)

        return candidates

    def _score_candidates(self, candidates: List[str]) -> List[GeneratedSequence]:
        """
        用漏斗感知策略评分候选序列
        
        对每条序列计算RF2评分、Final评分、组合评分,
        并估计P(RF2)、P(final|RF2)和P(final)。
        """
        scored = []

        for cdr3 in candidates:
            try:
                # 漏斗感知评分
                funnel_result = self.strategy.score_sequence_funnel(cdr3)

                # v2.4兼容软偏好评分
                soft_score = 0.0
                if self.base_strategy:
                    soft_score = score_soft_preferences(cdr3, self.base_strategy)

                # 估计概率 (sigmoid校准)
                p_rf2 = self._calibrate_probability(funnel_result.rf2_score, stage='rf2')
                p_final_given_rf2 = self._calibrate_probability(funnel_result.final_score, stage='final')
                p_final = p_rf2 * p_final_given_rf2

                gen_seq = GeneratedSequence(
                    cdr3=cdr3,
                    length=len(cdr3),
                    first_aa=cdr3[0],
                    rf2_score=funnel_result.rf2_score,
                    final_score=funnel_result.final_score,
                    combined_score=funnel_result.combined_score,
                    estimated_p_rf2=p_rf2,
                    estimated_p_final_given_rf2=p_final_given_rf2,
                    estimated_p_final=p_final,
                    soft_score=soft_score,
                    recommendation=funnel_result.recommendation,
                )
                scored.append(gen_seq)

            except Exception:
                continue

        return scored

    def _calibrate_probability(self, score: float, stage: str = 'rf2') -> float:
        """
        将评分校准为概率 (sigmoid映射)
        
        Args:
            score: 阶段评分
            stage: 阶段名称
            
        Returns:
            校准后的概率 [0, 1]
        """
        # 不同阶段的sigmoid参数
        if stage == 'rf2':
            # RF2评分范围约[-2, 8], 映射到[0, 0.5]
            midpoint = 3.0   # 评分=3时P(RF2)≈0.24 (基线)
            scale = 0.3
        elif stage == 'final':
            # Final评分范围约[-2, 2], 映射到[0, 0.1]
            midpoint = 0.0
            scale = 0.5
        else:
            midpoint = 0.0
            scale = 0.3

        prob = 1.0 / (1.0 + np.exp(-scale * (score - midpoint)))
        return float(np.clip(prob, 0.001, 0.999))

    def _diversity_filter(
        self,
        scored: List[GeneratedSequence],
        top_n: int,
    ) -> List[GeneratedSequence]:
        """
        多样性过滤: 确保选中序列之间的编辑距离 >= min_edit_distance
        
        贪心算法: 按P(final)降序遍历, 如果与已选序列的编辑距离都满足要求, 则选中。
        """
        if self.min_edit_distance <= 0 or len(scored) <= top_n:
            # 不需要多样性过滤
            selected = scored[:top_n]
            # 计算多样性评分
            for s in selected:
                s.diversity_score = self._compute_diversity(s.cdr3, [x.cdr3 for x in selected])
            return selected

        selected = []
        selected_seqs = []

        for candidate in scored:
            if len(selected) >= top_n:
                break

            # 检查与已选序列的编辑距离
            is_diverse = True
            for existing in selected_seqs:
                ed = self._edit_distance(candidate.cdr3, existing)
                if ed < self.min_edit_distance:
                    is_diverse = False
                    break

            if is_diverse:
                candidate.diversity_score = self._compute_diversity(
                    candidate.cdr3, selected_seqs
                )
                selected.append(candidate)
                selected_seqs.append(candidate.cdr3)

        return selected

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """
        计算两个字符串的Levenshtein编辑距离
        
        Args:
            s1: 字符串1
            s2: 字符串2
            
        Returns:
            编辑距离
        """
        m, n = len(s1), len(s2)
        if abs(m - n) > max(m, n):
            return max(m, n)

        # 优化: 短序列用简单实现
        if m * n <= 200:
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            return dp[m][n]

        # 长序列: 仅用两行
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
            prev = curr
        return prev[n]

    def _compute_diversity(self, seq: str, others: List[str]) -> float:
        """
        计算序列相对于其他序列的多样性评分
        
        定义为与最近邻序列的编辑距离
        """
        if not others:
            return 0.0
        distances = [self._edit_distance(seq, o) for o in others if o != seq]
        return float(np.mean(distances)) if distances else 0.0

    def generate_to_csv(
        self,
        output_path: Union[str, Path],
        n_samples: int = 10000,
        top_n: int = 500,
        **kwargs,
    ) -> pd.DataFrame:
        """
        生成序列并保存为CSV
        
        Args:
            output_path: 输出CSV路径
            n_samples: 采样数
            top_n: 返回数
            **kwargs: 传递给generate()的其他参数
            
        Returns:
            生成的DataFrame
        """
        sequences = self.generate(n_samples=n_samples, top_n=top_n, **kwargs)

        records = [s.to_dict() for s in sequences]
        df = pd.DataFrame(records)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"  输出: {output_path} ({len(df)} 条)")

        return df


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def run_funnel_generation(
    strategy_path: str = None,
    output_dir: Union[str, Path] = 'output_v3_funnel',
    n_samples: int = 10000,
    top_n: int = 500,
    min_edit_distance: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    运行漏斗感知序列生成的主入口函数
    
    Args:
        strategy_path: v3.x策略JSON文件路径
        output_dir: 输出目录
        n_samples: 采样候选数
        top_n: 返回的最大序列数
        min_edit_distance: 最小编辑距离
        seed: 随机种子
        verbose: 是否输出详细日志
        
    Returns:
        生成的DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化
    strategy = FunnelAwareStrategy(verbose=False)
    gen = FunnelAwareGenerator(
        strategy=strategy,
        base_strategy_path=strategy_path,
        min_edit_distance=min_edit_distance,
        verbose=verbose,
    )

    # 生成
    df = gen.generate_to_csv(
        output_path=output_dir / 'funnel_generated_sequences.csv',
        n_samples=n_samples,
        top_n=top_n,
        seed=seed,
    )

    # Shannon熵
    if not df.empty and 'sequence' in df.columns:
        seq_counts = Counter(df['sequence'])
        total = len(df)
        probs = [c / total for c in seq_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        if verbose:
            print(f"\n[生成统计]")
            print(f"  序列数: {len(df)}")
            print(f"  Shannon熵: {entropy:.4f}")
            print(f"  平均P(final): {df['predicted_p_final'].mean():.6f}")
            print(f"  首残基分布: {dict(df['first_aa'].value_counts().head(5))}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CSC-O Phase 4: 漏斗感知序列生成器')
    parser.add_argument('--strategy', default=None, help='v3.x策略JSON路径')
    parser.add_argument('--output', default='output_v3_funnel', help='输出目录')
    parser.add_argument('--n-samples', type=int, default=10000, help='采样候选数')
    parser.add_argument('--top-n', type=int, default=500, help='返回序列数')
    parser.add_argument('--min-edit-distance', type=int, default=3, help='最小编辑距离')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    run_funnel_generation(
        strategy_path=args.strategy,
        output_dir=args.output,
        n_samples=args.n_samples,
        top_n=args.top_n,
        min_edit_distance=args.min_edit_distance,
        seed=args.seed,
    )
