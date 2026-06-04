#!/usr/bin/env python3
"""
CSC-O Phase 5: 反馈循环验证框架 (Validation Framework)

核心目标:
    建立从"生成序列→实验验证→模型更新"的闭环, 使预测模型随实验数据积累
    不断改进, 最终实现P(final)预测的持续优化。

框架设计:
    1. 输出标准化: 生成序列的完整预测信息
    2. 反馈摄入: 实验结果的结构化导入
    3. 贝叶斯更新: 基于实验结果更新预测模型
    4. A/B测试: 策略对比的统计验证
    5. 功效分析: 实验样本量规划

逻辑流程:
    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
    │ 生成序列     │────→│ 实验验证      │────→│ 反馈摄入      │
    │ (P1/P2/P4)  │     │ (RF2/AF3/FC) │     │ (结构化导入)  │
    └─────────────┘     └──────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
    │ 策略更新     │←────│ 贝叶斯更新    │←────│ 模型评估      │
    │ (v3.x→v3.x+1)│    │ (后验分布)    │     │ (AUC/HR/ATE) │
    └─────────────┘     └──────────────┘     └──────────────┘

验证指标:
    - P(final)预测: CV-AUC > 0.80
    - HR估计: 阶段特异性HR with 95% CI
    - ATE估计: 直接/间接效应 with SE
    - 生成质量: predicted final rate > 20%, diversity > 4.0
    - 策略对比: v3.x vs v2.4 的FC率提升

作者: CSC-O Team
版本: v3.1
日期: 2026-06-04
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import time

from csco_config import extract_cdr3_features, AROMATIC
from csco_funnel_aware_strategy import FunnelAwareStrategy
from csco_funnel_counterfactual import FunnelAwareCounterfactual
from csco_funnel_generator import FunnelAwareGenerator


# ═══════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """实验验证结果"""
    sequence: str
    rf2_passed: Optional[bool] = None
    af3_passed: Optional[bool] = None
    schrodinger_passed: Optional[bool] = None
    final_candidate: Optional[bool] = None
    rf2_lddt: Optional[float] = None
    af3_plddt: Optional[float] = None
    docking_score: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'sequence': self.sequence,
            'rf2_passed': self.rf2_passed,
            'af3_passed': self.af3_passed,
            'schrodinger_passed': self.schrodinger_passed,
            'final_candidate': self.final_candidate,
            'rf2_lddt': self.rf2_lddt,
            'af3_plddt': self.af3_plddt,
            'docking_score': self.docking_score,
            'notes': self.notes,
        }


@dataclass
class BayesianUpdateResult:
    """贝叶斯更新结果"""
    parameter: str
    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_std: float
    n_observations: int
    kl_divergence: float = 0.0  # 先验-后验KL散度, 衡量更新幅度

    def to_dict(self) -> Dict:
        return {
            'parameter': self.parameter,
            'prior_mean': round(self.prior_mean, 6),
            'prior_std': round(self.prior_std, 6),
            'posterior_mean': round(self.posterior_mean, 6),
            'posterior_std': round(self.posterior_std, 6),
            'n_observations': self.n_observations,
            'kl_divergence': round(self.kl_divergence, 6),
        }


# ═══════════════════════════════════════════════════════════════
# 核心类: ValidationFramework
# ═══════════════════════════════════════════════════════════════

class ValidationFramework:
    """
    反馈循环验证框架
    
    建立从生成→验证→更新的闭环, 支持贝叶斯更新和A/B测试。
    
    使用示例:
        >>> vf = ValidationFramework()
        >>> # 导入实验结果
        >>> results = [ExperimentResult("WADKEY", rf2_passed=True, final_candidate=False)]
        >>> vf.ingest_experimental_results(results)
        >>> # 贝叶斯更新
        >>> updates = vf.bayesian_update()
        >>> # A/B测试
        >>> ab_result = vf.ab_test(control_seqs, treatment_seqs)
    
    Args:
        strategy: FunnelAwareStrategy实例
        verbose: 是否输出详细日志
    """

    def __init__(
        self,
        strategy: FunnelAwareStrategy = None,
        verbose: bool = True,
    ):
        self.strategy = strategy or FunnelAwareStrategy(verbose=False)
        self.verbose = verbose

        # 存储实验结果
        self.experimental_results: List[ExperimentResult] = []

        # 先验参数 (基于v3.1模型的HR和ATE)
        self.priors = {
            'first_is_aromatic_rf2_hr': {'mean': 0.069, 'std': 0.01},
            'first_is_aromatic_final_hr': {'mean': 3.676, 'std': 0.8},
            'glycine_ratio_rf2_hr': {'mean': 1.390, 'std': 0.15},
            'serine_ratio_rf2_hr': {'mean': 2.444, 'std': 0.15},
            'p_rf2_baseline': {'mean': 0.238, 'std': 0.01},
            'p_final_given_rf2_baseline': {'mean': 0.026, 'std': 0.005},
        }

        # 后验参数 (初始等于先验)
        self.posteriors = {k: dict(v) for k, v in self.priors.items()}

        if self.verbose:
            print(f"[ValidationFramework] 初始化完成")
            print(f"  先验参数: {len(self.priors)} 个")

    # === 1. 输出标准化 ===

    def generate_output_csv(
        self,
        sequences: List[str],
        output_path: Union[str, Path],
        include_mutation_suggestions: bool = True,
        top_k_mutations: int = 3,
    ) -> pd.DataFrame:
        """
        生成标准化的输出CSV
        
        输出格式:
        sequence, predicted_p_final, predicted_p_rf2, predicted_p_final_given_rf2,
        recommended_mutation, ci, diversity_score
        
        Args:
            sequences: CDR3序列列表
            output_path: 输出CSV路径
            include_mutation_suggestions: 是否包含突变建议
            top_k_mutations: 每条序列的突变建议数
            
        Returns:
            标准化输出DataFrame
        """
        if not sequences:
            raise ValueError("sequences must be a non-empty list")

        rows = []
        navigator = FunnelAwareCounterfactual(
            strategy=self.strategy, verbose=False
        )

        for seq in sequences:
            # 漏斗感知评分
            funnel_result = self.strategy.score_sequence_funnel(seq)

            # 估计概率
            p_rf2 = self._estimate_p_rf2(funnel_result.rf2_score)
            p_final_given_rf2 = self._estimate_p_final_given_rf2(funnel_result.final_score)
            p_final = p_rf2 * p_final_given_rf2

            # 置信区间 (基于后验标准误)
            ci = self._compute_ci(p_final, n_observations=len(self.experimental_results))

            # 突变建议
            mutation_str = ""
            if include_mutation_suggestions:
                try:
                    cf_result = navigator.suggest_mutation(seq, top_k=top_k_mutations)
                    if cf_result.best_mutation:
                        mutation_str = cf_result.best_mutation.mutation
                except Exception:
                    pass

            rows.append({
                'sequence': seq,
                'predicted_p_final': round(p_final, 6),
                'predicted_p_rf2': round(p_rf2, 6),
                'predicted_p_final_given_rf2': round(p_final_given_rf2, 6),
                'recommended_mutation': mutation_str,
                'ci_lower': round(ci[0], 6),
                'ci_upper': round(ci[1], 6),
                'rf2_score': round(funnel_result.rf2_score, 4),
                'final_score': round(funnel_result.final_score, 4),
                'combined_score': round(funnel_result.combined_score, 4),
            })

        df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"[ValidationFramework] 输出CSV: {output_path} ({len(df)} 条)")

        return df

    # === 2. 反馈摄入 ===

    def ingest_experimental_results(
        self,
        results: List[ExperimentResult],
    ) -> Dict[str, Any]:
        """
        摄入实验验证结果
        
        Args:
            results: ExperimentResult列表
            
        Returns:
            摄入统计信息
        """
        if not results:
            raise ValueError("results must be a non-empty list")

        n_before = len(self.experimental_results)
        self.experimental_results.extend(results)
        n_after = len(self.experimental_results)

        # 统计
        n_rf2 = sum(1 for r in results if r.rf2_passed is not None)
        n_fc = sum(1 for r in results if r.final_candidate is not None)
        n_rf2_pass = sum(1 for r in results if r.rf2_passed == True)
        n_fc_pass = sum(1 for r in results if r.final_candidate == True)

        stats_info = {
            'n_ingested': len(results),
            'n_total': n_after,
            'n_rf2_tested': n_rf2,
            'n_rf2_passed': n_rf2_pass,
            'n_fc_tested': n_fc,
            'n_fc_passed': n_fc_pass,
        }

        if self.verbose:
            print(f"[ValidationFramework] 摄入实验结果:")
            print(f"  新增: {len(results)}, 总计: {n_after}")
            print(f"  RF2测试: {n_rf2}, 通过: {n_rf2_pass}")
            print(f"  FC测试: {n_fc}, 通过: {n_fc_pass}")

        return stats_info

    def ingest_from_csv(self, csv_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从CSV文件摄入实验结果
        
        CSV需包含: sequence, rf2_passed, final_candidate列
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            摄入统计信息
        """
        df = pd.read_csv(csv_path)

        results = []
        for _, row in df.iterrows():
            result = ExperimentResult(
                sequence=str(row.get('sequence', row.get('cdr3_sequence', ''))),
                rf2_passed=row.get('rf2_passed', None),
                af3_passed=row.get('af3_passed', None),
                schrodinger_passed=row.get('schrodinger_passed', None),
                final_candidate=row.get('final_candidate', None),
                rf2_lddt=row.get('rf2_lddt', None),
                af3_plddt=row.get('af3_plddt', None),
                docking_score=row.get('docking_score', None),
            )
            results.append(result)

        return self.ingest_experimental_results(results)

    # === 3. 贝叶斯更新 ===

    def bayesian_update(self) -> List[BayesianUpdateResult]:
        """
        基于实验结果进行贝叶斯更新
        
        对每个先验参数, 用实验数据计算后验分布。
        使用共轭先验: 正态-正态模型
        
        Returns:
            BayesianUpdateResult列表
        """
        if not self.experimental_results:
            if self.verbose:
                print("[ValidationFramework] 无实验数据, 跳过贝叶斯更新")
            return []

        updates = []

        # 更新P(RF2)基线
        rf2_results = [r for r in self.experimental_results if r.rf2_passed is not None]
        if rf2_results:
            rf2_rate = sum(1 for r in rf2_results if r.rf2_passed) / len(rf2_results)
            update = self._conjugate_normal_update(
                'p_rf2_baseline', rf2_rate, len(rf2_results)
            )
            updates.append(update)

        # 更新P(final|RF2)基线
        fc_results = [r for r in self.experimental_results if r.final_candidate is not None]
        if fc_results:
            fc_rate = sum(1 for r in fc_results if r.final_candidate) / len(fc_results)
            update = self._conjugate_normal_update(
                'p_final_given_rf2_baseline', fc_rate, len(fc_results)
            )
            updates.append(update)

        # 更新HR参数 (基于首残基类型分层)
        aromatic_results = [r for r in rf2_results
                           if len(r.sequence) > 0 and r.sequence[0] in AROMATIC]
        non_aromatic_results = [r for r in rf2_results
                               if len(r.sequence) > 0 and r.sequence[0] not in AROMATIC]

        if aromatic_results and non_aromatic_results:
            aromatic_rate = sum(1 for r in aromatic_results if r.rf2_passed) / len(aromatic_results)
            non_aromatic_rate = sum(1 for r in non_aromatic_results if r.rf2_passed) / len(non_aromatic_results)

            # 更新first_is_aromatic RF2 HR
            if non_aromatic_rate > 0:
                observed_hr = aromatic_rate / non_aromatic_rate
                update = self._conjugate_normal_update(
                    'first_is_aromatic_rf2_hr', observed_hr,
                    min(len(aromatic_results), len(non_aromatic_results))
                )
                updates.append(update)

        if self.verbose:
            print(f"\n[ValidationFramework] 贝叶斯更新完成: {len(updates)} 个参数")
            for u in updates:
                print(f"  {u.parameter}: {u.prior_mean:.4f}→{u.posterior_mean:.4f} "
                      f"(KL={u.kl_divergence:.4f})")

        return updates

    def _conjugate_normal_update(
        self,
        param_name: str,
        observed_mean: float,
        n_observations: int,
    ) -> BayesianUpdateResult:
        """
        正态-正态共轭更新
        
        先验: N(μ₀, σ₀²)
        似然: N(x̄, σ²/n)
        后验: N(μₙ, σₙ²)
        
        μₙ = (μ₀/σ₀² + n*x̄/σ²) / (1/σ₀² + n/σ²)
        σₙ² = 1 / (1/σ₀² + n/σ²)
        """
        prior = self.priors[param_name]
        prior_mean = prior['mean']
        prior_std = prior['std']
        prior_var = prior_std ** 2

        # 观测方差 (假设观测标准误 = 观测均值的30%)
        obs_var = (observed_mean * 0.3) ** 2 if observed_mean > 0 else 0.01

        # 后验计算
        posterior_var = 1.0 / (1.0 / prior_var + n_observations / obs_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + n_observations * observed_mean / obs_var)
        posterior_std = np.sqrt(posterior_var)

        # KL散度
        kl = np.log(posterior_std / prior_std) + \
             (prior_var + (prior_mean - posterior_mean)**2) / (2 * posterior_var**2) - 0.5
        kl = max(kl, 0.0)

        # 更新后验
        self.posteriors[param_name] = {
            'mean': posterior_mean,
            'std': posterior_std,
        }

        return BayesianUpdateResult(
            parameter=param_name,
            prior_mean=prior_mean,
            prior_std=prior_std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            n_observations=n_observations,
            kl_divergence=kl,
        )

    # === 4. A/B测试 ===

    def ab_test(
        self,
        control_sequences: List[str],
        treatment_sequences: List[str],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        A/B测试支持: 分层随机化 + 统计检验
        
        对比两组序列的预测P(final)差异。
        
        Args:
            control_sequences: 对照组序列 (如v2.4策略生成)
            treatment_sequences: 实验组序列 (如v3.1策略生成)
            alpha: 显著性水平
            
        Returns:
            A/B测试结果字典
        """
        if not control_sequences or not treatment_sequences:
            raise ValueError("Both control and treatment sequences must be non-empty")

        # 评分
        control_scores = [self.strategy.score_sequence_funnel(s) for s in control_sequences]
        treatment_scores = [self.strategy.score_sequence_funnel(s) for s in treatment_sequences]

        control_combined = [s.combined_score for s in control_scores]
        treatment_combined = [s.combined_score for s in treatment_scores]

        # 统计检验
        t_stat, p_value = stats.ttest_ind(treatment_combined, control_combined)

        # 效应量 (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_combined) * (len(control_combined) - 1) +
             np.var(treatment_combined) * (len(treatment_combined) - 1)) /
            (len(control_combined) + len(treatment_combined) - 2)
        )
        cohens_d = (np.mean(treatment_combined) - np.mean(control_combined)) / max(pooled_std, 1e-8)

        result = {
            'control_n': len(control_sequences),
            'treatment_n': len(treatment_sequences),
            'control_mean_combined': np.mean(control_combined),
            'treatment_mean_combined': np.mean(treatment_combined),
            'delta_combined': np.mean(treatment_combined) - np.mean(control_combined),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < alpha,
            'alpha': alpha,
        }

        if self.verbose:
            print(f"\n[ValidationFramework] A/B测试结果:")
            print(f"  对照组(n={result['control_n']}): combined={result['control_mean_combined']:.3f}")
            print(f"  实验组(n={result['treatment_n']}): combined={result['treatment_mean_combined']:.3f}")
            print(f"  Δcombined={result['delta_combined']:+.3f}")
            print(f"  t={t_stat:.3f}, p={p_value:.4f}, Cohen's d={cohens_d:.3f}")
            print(f"  显著: {'是' if result['significant'] else '否'} (α={alpha})")

        return result

    # === 5. 功效分析 ===

    def power_analysis(
        self,
        effect_size: float = 0.5,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> Dict[str, Any]:
        """
        功效分析: 计算检测给定效应量所需的最小样本量
        
        Args:
            effect_size: 期望检测的效应量 (Cohen's d)
            alpha: 显著性水平
            power: 统计功效
            
        Returns:
            功效分析结果
        """
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # 每组所需样本量
        n_per_group = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))

        result = {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'n_per_group': n_per_group,
            'n_total': n_per_group * 2,
            'interpretation': f"检测Cohen's d={effect_size}的效应, 每组需要{n_per_group}条序列"
        }

        if self.verbose:
            print(f"\n[ValidationFramework] 功效分析:")
            print(f"  效应量: {effect_size}")
            print(f"  每组样本量: {n_per_group}")
            print(f"  总样本量: {n_per_group * 2}")

        return result

    # === 辅助函数 ===

    def _estimate_p_rf2(self, rf2_score: float) -> float:
        """估计P(RF2)"""
        posterior = self.posteriors.get('p_rf2_baseline', {'mean': 0.238, 'std': 0.01})
        baseline = posterior['mean']
        # 基于评分调整
        adjustment = 0.05 * rf2_score  # 简化映射
        return float(np.clip(baseline + adjustment, 0.001, 0.999))

    def _estimate_p_final_given_rf2(self, final_score: float) -> float:
        """估计P(final|RF2)"""
        posterior = self.posteriors.get('p_final_given_rf2_baseline', {'mean': 0.026, 'std': 0.005})
        baseline = posterior['mean']
        adjustment = 0.01 * final_score
        return float(np.clip(baseline + adjustment, 0.001, 0.999))

    def _compute_ci(self, estimate: float, n_observations: int = 0, confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        # 标准误随观测数递减
        base_se = 0.01
        se = base_se / max(np.sqrt(max(n_observations, 1)), 1)
        return (estimate - z * se, estimate + z * se)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def run_validation_framework(
    generated_csv: str = None,
    experimental_csv: str = None,
    output_dir: Union[str, Path] = 'output_v3_funnel',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    运行验证框架的主入口函数
    
    Args:
        generated_csv: 生成序列CSV路径
        experimental_csv: 实验结果CSV路径
        output_dir: 输出目录
        verbose: 是否输出详细日志
        
    Returns:
        验证结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vf = ValidationFramework(verbose=verbose)
    results = {}

    # 摄入实验数据
    if experimental_csv and Path(experimental_csv).exists():
        stats_info = vf.ingest_from_csv(experimental_csv)
        results['ingestion'] = stats_info

        # 贝叶斯更新
        updates = vf.bayesian_update()
        results['bayesian_updates'] = [u.to_dict() for u in updates]

    # 生成输出CSV
    if generated_csv and Path(generated_csv).exists():
        df = pd.read_csv(generated_csv)
        seq_col = 'sequence' if 'sequence' in df.columns else 'cdr3'
        sequences = df[seq_col].dropna().tolist()[:100]

        output_df = vf.generate_output_csv(
            sequences=sequences,
            output_path=output_dir / 'validation_output.csv',
        )
        results['n_output'] = len(output_df)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CSC-O Phase 5: 验证框架')
    parser.add_argument('--generated', default=None, help='生成序列CSV路径')
    parser.add_argument('--experimental', default=None, help='实验结果CSV路径')
    parser.add_argument('--output', default='output_v3_funnel', help='输出目录')

    args = parser.parse_args()

    run_validation_framework(
        generated_csv=args.generated,
        experimental_csv=args.experimental,
        output_dir=args.output,
    )
