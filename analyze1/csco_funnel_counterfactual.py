#!/usr/bin/env python3
"""
CSC-O Phase 3: 漏斗感知反事实导航 (Funnel-Aware Counterfactual Navigator)

核心功能:
    基于P(final)而非P(RF2)生成突变建议。对每条序列，枚举所有单点突变，
    计算每个突变对P(RF2)、P(final|RF2)和P(final)的影响，返回使P(final)
    最大化且满足阶段特异性约束的top-k突变建议。

设计原则:
    - 分解建模: P(final) = P(RF2) × P(final|RF2)
    - 阶段感知: 区分RF2阶段效应和Final阶段效应
    - 不确定性量化: 输出置信区间和证据强度
    - 可扩展: method参数预留transfer/direct方法

依赖:
    - csco_multistage_causal.py: MultiStageMediationModel
    - csco_multistate_survival.py: MultiStateSurvivalModel
    - csco_funnel_aware_strategy.py: FunnelAwareStrategy
    - csco_config.py: extract_cdr3_features, AMINO_ACIDS, AROMATIC

作者: CSC-O Team
版本: v3.1
日期: 2026-06-04
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import time

from csco_config import (
    AMINO_ACIDS, AROMATIC, GLYCINE, SERINE, PROLINE, HYDROPHOBIC,
    POSITIVE, NEGATIVE,
    extract_cdr3_features,
)
from csco_funnel_aware_strategy import (
    FunnelAwareStrategy,
    DEFAULT_STAGE_AWARE_CONSTRAINTS,
    DEFAULT_FUNNEL_WEIGHTS,
)


# ═══════════════════════════════════════════════════════════════
# 数据类定义
# ═══════════════════════════════════════════════════════════════

@dataclass
class MutationSuggestion:
    """
    单点突变建议
    
    Attributes:
        mutation: 突变描述, 格式 "原残基→新残基@位置" (如 "G→F@pos0")
        position: 突变位置 (0-indexed)
        original_aa: 原始氨基酸
        new_aa: 新氨基酸
        mutated_sequence: 突变后的完整序列
        delta_p_rf2: 对P(RF2)的影响 (百分点)
        delta_p_final_given_rf2: 对P(final|RF2)的影响 (百分点)
        delta_p_final: 对P(final)的总影响 (百分点)
        rf2_score_delta: RF2阶段评分变化
        final_score_delta: Final阶段评分变化
        combined_score_delta: 组合评分变化
        confidence: 置信度 ('high'/'medium'/'low')
        evidence: 支撑证据描述
        stage_breakdown: 阶段特异性效应分解
    """
    mutation: str
    position: int
    original_aa: str
    new_aa: str
    mutated_sequence: str
    delta_p_rf2: float = 0.0
    delta_p_final_given_rf2: float = 0.0
    delta_p_final: float = 0.0
    rf2_score_delta: float = 0.0
    final_score_delta: float = 0.0
    combined_score_delta: float = 0.0
    confidence: str = 'medium'
    evidence: str = ""
    stage_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'mutation': self.mutation,
            'position': self.position,
            'original_aa': self.original_aa,
            'new_aa': self.new_aa,
            'mutated_sequence': self.mutated_sequence,
            'delta_p_rf2': round(self.delta_p_rf2, 4),
            'delta_p_final_given_rf2': round(self.delta_p_final_given_rf2, 4),
            'delta_p_final': round(self.delta_p_final, 4),
            'rf2_score_delta': round(self.rf2_score_delta, 4),
            'final_score_delta': round(self.final_score_delta, 4),
            'combined_score_delta': round(self.combined_score_delta, 4),
            'confidence': self.confidence,
            'evidence': self.evidence,
        }


@dataclass
class CounterfactualResult:
    """
    反事实分析结果
    
    Attributes:
        original_sequence: 原始序列
        original_scores: 原始序列的阶段评分
        suggestions: 突变建议列表 (按delta_p_final降序)
        n_mutations_evaluated: 评估的突变总数
        n_beneficial: 有益突变数 (delta_p_final > 0)
        best_mutation: 最优突变建议
    """
    original_sequence: str
    original_scores: Dict[str, float]
    suggestions: List[MutationSuggestion]
    n_mutations_evaluated: int = 0
    n_beneficial: int = 0
    best_mutation: Optional[MutationSuggestion] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_sequence': self.original_sequence,
            'original_rf2_score': self.original_scores.get('rf2_score', 0),
            'original_final_score': self.original_scores.get('final_score', 0),
            'original_combined_score': self.original_scores.get('combined_score', 0),
            'n_mutations_evaluated': self.n_mutations_evaluated,
            'n_beneficial': self.n_beneficial,
            'best_mutation': self.best_mutation.to_dict() if self.best_mutation else None,
            'top_suggestions': [s.to_dict() for s in self.suggestions],
        }


# ═══════════════════════════════════════════════════════════════
# 核心类: FunnelAwareCounterfactual
# ═══════════════════════════════════════════════════════════════

class FunnelAwareCounterfactual:
    """
    漏斗感知反事实导航器
    
    基于P(final)而非P(RF2)生成突变建议。对每条序列枚举所有单点突变，
    用漏斗感知策略评分每个突变，返回使P(final)最大化的top-k建议。
    
    使用示例:
        >>> from csco_funnel_aware_strategy import FunnelAwareStrategy
        >>> strategy = FunnelAwareStrategy()
        >>> navigator = FunnelAwareCounterfactual(strategy)
        >>> result = navigator.suggest_mutation("WADKEY", top_k=5)
        >>> for s in result.suggestions:
        ...     print(f"{s.mutation}: ΔP(final)={s.delta_p_final:+.2f}")
    
    Args:
        strategy: FunnelAwareStrategy实例, 提供阶段感知评分
        method: 估计方法, 'decomposition'(默认)/'transfer'/'direct'
        available_stages: 可用阶段列表
        verbose: 是否输出详细日志
    """

    def __init__(
        self,
        strategy: FunnelAwareStrategy = None,
        method: str = 'decomposition',
        available_stages: List[str] = None,
        verbose: bool = True,
    ):
        # 参数验证
        if method not in ('decomposition', 'transfer', 'direct'):
            raise ValueError(f"method must be 'decomposition'/'transfer'/'direct', got '{method}'")
        
        self.strategy = strategy or FunnelAwareStrategy(verbose=False)
        self.method = method
        self.available_stages = available_stages or ['rf2', 'final']
        self.verbose = verbose

        # 阶段权重
        self.funnel_weights = self.strategy.funnel_weights

        if self.verbose:
            print(f"[FunnelAwareCounterfactual] 初始化完成")
            print(f"  估计方法: {self.method}")
            print(f"  阶段权重: {self.funnel_weights}")

    def suggest_mutation(
        self,
        sequence: str,
        top_k: int = 5,
        positions: List[int] = None,
        allowed_aa: List[str] = None,
        min_delta_p_final: float = 0.0,
    ) -> CounterfactualResult:
        """
        对单条序列生成漏斗感知突变建议
        
        枚举所有单点突变, 计算每个突变对P(RF2)、P(final|RF2)和P(final)
        的影响, 返回使P(final)最大化的top-k建议。
        
        Args:
            sequence: CDR3序列字符串, 长度需在5-13之间
            top_k: 返回的最大建议数, 需为正整数
            positions: 限制突变位置列表 (0-indexed), None表示所有位置
            allowed_aa: 允许的替换氨基酸列表, None表示全部20种
            min_delta_p_final: 最小P(final)提升阈值, 低于此值的建议被过滤
            
        Returns:
            CounterfactualResult对象, 包含原始评分和排序后的突变建议
            
        Raises:
            ValueError: 序列为空、长度超范围或参数类型错误
        """
        # === 输入验证 ===
        self._validate_sequence(sequence)
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if min_delta_p_final < 0:
            raise ValueError(f"min_delta_p_final must be non-negative, got {min_delta_p_final}")

        t0 = time.time()

        # === 原始序列评分 ===
        original_result = self.strategy.score_sequence_funnel(sequence)
        original_scores = {
            'rf2_score': original_result.rf2_score,
            'final_score': original_result.final_score,
            'combined_score': original_result.combined_score,
        }

        if self.verbose:
            print(f"\n[FunnelAwareCounterfactual] 分析序列: {sequence}")
            print(f"  原始评分: RF2={original_scores['rf2_score']:.3f}, "
                  f"Final={original_scores['final_score']:.3f}, "
                  f"Combined={original_scores['combined_score']:.3f}")

        # === 枚举突变 ===
        seq_len = len(sequence)
        target_positions = positions if positions is not None else list(range(seq_len))
        target_aa = allowed_aa if allowed_aa is not None else list(AMINO_ACIDS)

        suggestions = []
        n_evaluated = 0

        for pos in target_positions:
            if pos < 0 or pos >= seq_len:
                continue

            original_aa = sequence[pos]

            for new_aa in target_aa:
                # 跳过无变化
                if new_aa == original_aa:
                    continue

                # 构建突变序列
                mutated = sequence[:pos] + new_aa + sequence[pos + 1:]

                # 评估突变
                try:
                    suggestion = self._evaluate_mutation(
                        original_seq=sequence,
                        mutated_seq=mutated,
                        position=pos,
                        original_aa=original_aa,
                        new_aa=new_aa,
                        original_scores=original_scores,
                    )
                    n_evaluated += 1

                    # 过滤低效突变
                    if suggestion.delta_p_final >= min_delta_p_final:
                        suggestions.append(suggestion)

                except Exception as e:
                    if self.verbose:
                        print(f"  [WARN] 突变 {original_aa}→{new_aa}@pos{pos} 评估失败: {e}")
                    continue

        # === 排序 ===
        suggestions.sort(key=lambda s: -s.delta_p_final)
        top_suggestions = suggestions[:top_k]

        # === 统计 ===
        n_beneficial = sum(1 for s in suggestions if s.delta_p_final > 0)
        best = top_suggestions[0] if top_suggestions else None

        elapsed = time.time() - t0
        if self.verbose:
            print(f"  评估突变: {n_evaluated}, 有益: {n_beneficial}")
            if best:
                print(f"  最优: {best.mutation} ΔP(final)={best.delta_p_final:+.4f} "
                      f"[RF2:{best.delta_p_rf2:+.4f}, Final:{best.delta_p_final_given_rf2:+.4f}]")
            print(f"  耗时: {elapsed:.2f}s")

        return CounterfactualResult(
            original_sequence=sequence,
            original_scores=original_scores,
            suggestions=top_suggestions,
            n_mutations_evaluated=n_evaluated,
            n_beneficial=n_beneficial,
            best_mutation=best,
        )

    def suggest_mutation_batch(
        self,
        sequences: List[str],
        top_k: int = 5,
        min_delta_p_final: float = 0.0,
    ) -> Dict[str, CounterfactualResult]:
        """
        批量生成突变建议
        
        Args:
            sequences: CDR3序列列表
            top_k: 每条序列返回的最大建议数
            min_delta_p_final: 最小P(final)提升阈值
            
        Returns:
            字典: 序列 → CounterfactualResult
            
        Raises:
            ValueError: sequences为空或包含无效序列
        """
        if not sequences:
            raise ValueError("sequences must be a non-empty list")
        if not isinstance(sequences, list):
            raise ValueError(f"sequences must be a list, got {type(sequences).__name__}")

        results = {}
        for i, seq in enumerate(sequences):
            try:
                result = self.suggest_mutation(
                    sequence=seq,
                    top_k=top_k,
                    min_delta_p_final=min_delta_p_final,
                )
                results[seq] = result
            except ValueError as e:
                if self.verbose:
                    print(f"  [SKIP] 序列#{i} '{seq}' 无效: {e}")
                continue

        if self.verbose:
            print(f"\n[Batch] 完成: {len(results)}/{len(sequences)} 条序列成功分析")

        return results

    def _evaluate_mutation(
        self,
        original_seq: str,
        mutated_seq: str,
        position: int,
        original_aa: str,
        new_aa: str,
        original_scores: Dict[str, float],
    ) -> MutationSuggestion:
        """
        评估单个突变的效果
        
        通过比较原始序列和突变序列的漏斗感知评分, 计算突变对P(final)的影响。
        
        Args:
            original_seq: 原始序列
            mutated_seq: 突变后序列
            position: 突变位置
            original_aa: 原始氨基酸
            new_aa: 新氨基酸
            original_scores: 原始序列的阶段评分
            
        Returns:
            MutationSuggestion对象
        """
        # 突变后评分
        mutated_result = self.strategy.score_sequence_funnel(mutated_seq)

        # 评分变化
        rf2_delta = mutated_result.rf2_score - original_scores['rf2_score']
        final_delta = mutated_result.final_score - original_scores['final_score']
        combined_delta = mutated_result.combined_score - original_scores['combined_score']

        # 将评分变化映射为概率变化
        # 使用校准因子: 基于历史数据的评分-概率关系
        delta_p_rf2 = self._score_to_probability_delta(rf2_delta, stage='rf2')
        delta_p_final_given_rf2 = self._score_to_probability_delta(final_delta, stage='final')
        # P(final) = P(RF2) × P(final|RF2), ΔP(final)由组合评分直接映射
        delta_p_final = self._score_to_probability_delta(combined_delta, stage='combined')

        # 置信度评估
        confidence = self._assess_confidence(position, original_aa, new_aa, original_seq)

        # 证据生成
        evidence = self._generate_evidence(position, original_aa, new_aa, rf2_delta, final_delta)

        # 阶段分解
        stage_breakdown = {
            'rf2_score_delta': rf2_delta,
            'final_score_delta': final_delta,
            'rf2_contribution': delta_p_rf2,
            'final_contribution': delta_p_final_given_rf2,
        }

        return MutationSuggestion(
            mutation=f"{original_aa}→{new_aa}@pos{position}",
            position=position,
            original_aa=original_aa,
            new_aa=new_aa,
            mutated_sequence=mutated_seq,
            delta_p_rf2=delta_p_rf2,
            delta_p_final_given_rf2=delta_p_final_given_rf2,
            delta_p_final=delta_p_final,
            rf2_score_delta=rf2_delta,
            final_score_delta=final_delta,
            combined_score_delta=combined_delta,
            confidence=confidence,
            evidence=evidence,
            stage_breakdown=stage_breakdown,
        )

    def _score_to_probability_delta(self, score_delta: float, stage: str = 'rf2') -> float:
        """
        将评分变化映射为概率变化
        
        使用sigmoid校准: ΔP = sigmoid(score + Δscore) - sigmoid(score)
        简化: ΔP ≈ Δscore × calibration_factor
        
        Args:
            score_delta: 评分变化量
            stage: 阶段名称 ('rf2' 或 'final')
            
        Returns:
            概率变化量 (百分点)
        """
        # 校准因子: 基于历史数据拟合
        # RF2阶段: 评分范围约[-2, 8], P(RF2)范围约[0, 0.5]
        # Final阶段: 评分范围约[-2, 2], P(final|RF2)范围约[0, 0.1]
        if stage == 'rf2':
            calibration = 0.05  # 1分 ≈ 5pp P(RF2)变化
        elif stage == 'final':
            calibration = 0.02  # 1分 ≈ 2pp P(final|RF2)变化
        elif stage == 'combined':
            calibration = 0.03  # 1分 ≈ 3pp P(final)变化
        else:
            calibration = 0.01

        return score_delta * calibration

    def _assess_confidence(
        self,
        position: int,
        original_aa: str,
        new_aa: str,
        sequence: str,
    ) -> str:
        """
        评估突变建议的置信度
        
        基于以下因素:
        - 位置0(首残基): 有强因果证据, 置信度高
        - 甘氨酸/丝氨酸比例变化: 有Cox HR证据, 置信度中-高
        - 其他位置: 证据较弱, 置信度低
        
        Args:
            position: 突变位置
            original_aa: 原始氨基酸
            new_aa: 新氨基酸
            sequence: 原始序列
            
        Returns:
            'high'/'medium'/'low'
        """
        # 首残基突变: 有强因果证据 (HR, ATE)
        if position == 0:
            if (original_aa in AROMATIC) != (new_aa in AROMATIC):
                return 'high'  # 芳香族↔非芳香族切换
            return 'medium'

        # 甘氨酸相关突变
        if original_aa in GLYCINE or new_aa in GLYCINE:
            return 'medium'  # 有Cox HR证据

        # 丝氨酸相关突变
        if original_aa in SERINE or new_aa in SERINE:
            return 'medium'

        # 其他位置: 证据较弱
        return 'low'

    def _generate_evidence(
        self,
        position: int,
        original_aa: str,
        new_aa: str,
        rf2_delta: float,
        final_delta: float,
    ) -> str:
        """
        生成突变建议的支撑证据描述
        
        Args:
            position: 突变位置
            original_aa: 原始氨基酸
            new_aa: 新氨基酸
            rf2_delta: RF2评分变化
            final_delta: Final评分变化
            
        Returns:
            证据描述字符串
        """
        parts = []

        # 首残基证据
        if position == 0:
            if original_aa in AROMATIC and new_aa not in AROMATIC:
                parts.append("芳香族→非芳香族: RF2 HR=0.069(保护)但Final HR=3.676(风险反转)")
            elif original_aa not in AROMATIC and new_aa in AROMATIC:
                parts.append("非芳香族→芳香族: RF2保护(HR=0.069)但Final风险(HR=3.676)")

            if new_aa == 'G':
                parts.append("G首残基: FC正样本中占26.2%, 需glycine_ratio≤0.15")

        # 甘氨酸证据
        if new_aa in GLYCINE and position > 0:
            parts.append("引入甘氨酸: RF2风险(HR=1.39)")
        elif original_aa in GLYCINE and new_aa not in GLYCINE and position > 0:
            parts.append("移除甘氨酸: 降低RF2风险")

        # 丝氨酸证据
        if new_aa in SERINE and position > 0:
            parts.append("引入丝氨酸: RF2强风险(HR=2.44)")
        elif original_aa in SERINE and new_aa not in SERINE and position > 0:
            parts.append("移除丝氨酸: 降低RF2强风险")

        # 阶段效应方向
        if rf2_delta > 0 and final_delta < 0:
            parts.append("阶段冲突: RF2有利但Final不利")
        elif rf2_delta < 0 and final_delta > 0:
            parts.append("阶段冲突: Final有利但RF2不利")
        elif rf2_delta > 0 and final_delta > 0:
            parts.append("阶段协同: 两阶段均有利")

        return "; ".join(parts) if parts else "基于漏斗感知评分"

    def _validate_sequence(self, sequence: str):
        """
        验证CDR3序列的有效性
        
        Args:
            sequence: CDR3序列
            
        Raises:
            ValueError: 序列无效
        """
        if not isinstance(sequence, str):
            raise ValueError(f"sequence must be a string, got {type(sequence).__name__}")
        if len(sequence) == 0:
            raise ValueError("sequence must not be empty")
        if len(sequence) < 5 or len(sequence) > 13:
            raise ValueError(f"sequence length must be 5-13, got {len(sequence)}")
        invalid_chars = set(sequence) - set(AMINO_ACIDS)
        if invalid_chars:
            raise ValueError(f"sequence contains invalid amino acids: {invalid_chars}")


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def run_counterfactual_analysis(
    sequences: List[str],
    output_dir: Union[str, Path],
    top_k: int = 5,
    method: str = 'decomposition',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    运行反事实分析的主入口函数
    
    Args:
        sequences: 待分析的CDR3序列列表
        output_dir: 输出目录
        top_k: 每条序列返回的最大建议数
        method: 估计方法
        verbose: 是否输出详细日志
        
    Returns:
        汇总DataFrame, 包含所有序列的突变建议
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化
    strategy = FunnelAwareStrategy(verbose=False)
    navigator = FunnelAwareCounterfactual(
        strategy=strategy,
        method=method,
        verbose=verbose,
    )

    # 批量分析
    results = navigator.suggest_mutation_batch(sequences, top_k=top_k)

    # 汇总
    rows = []
    for seq, result in results.items():
        for suggestion in result.suggestions:
            row = suggestion.to_dict()
            row['original_sequence'] = seq
            row['original_rf2_score'] = result.original_scores.get('rf2_score', 0)
            row['original_final_score'] = result.original_scores.get('final_score', 0)
            row['original_combined_score'] = result.original_scores.get('combined_score', 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / 'counterfactual_suggestions.csv', index=False)

    if verbose:
        print(f"\n[Counterfactual] 完成: {len(results)} 条序列, {len(df)} 条建议")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CSC-O Phase 3: 漏斗感知反事实导航')
    parser.add_argument('--input', required=True, help='输入CSV文件路径 (需含cdr3_sequence列)')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--top-k', type=int, default=5, help='每条序列返回的最大建议数')
    parser.add_argument('--method', default='decomposition',
                        choices=['decomposition', 'transfer', 'direct'], help='估计方法')
    parser.add_argument('--max-sequences', type=int, default=100, help='最大分析序列数')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 加载数据
    df = pd.read_csv(args.input)
    seq_col = 'cdr3_sequence' if 'cdr3_sequence' in df.columns else 'cdr3'
    if seq_col not in df.columns:
        print(f"错误: 找不到CDR3序列列 (尝试了 'cdr3_sequence' 和 'cdr3')")
        sys.exit(1)

    sequences = df[seq_col].dropna().tolist()[:args.max_sequences]

    # 运行分析
    result_df = run_counterfactual_analysis(
        sequences=sequences,
        output_dir=args.output,
        top_k=args.top_k,
        method=args.method,
        verbose=args.verbose,
    )

    print(f"\n完成! 共生成 {len(result_df)} 条突变建议")
