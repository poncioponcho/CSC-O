#!/usr/bin/env python3
"""
CSC-O Funnel-Aware Strategy: 基于HR反转发现的阶段感知生成策略

核心发现:
  - first_is_aromatic: RF2 HR=0.069 (极强保护) → Final HR=3.676 [2.42-5.58] (风险反转, p=1e-9)
  - cdr3_length_bin:  RF2 HR=0.096 (强保护)   → Final HR=1.928 [1.20-3.11] (风险反转, p=0.007)
  - glycine_ratio_bin: RF2 HR=1.390 (风险)    → Final HR=0.863 [0.54-1.39] (不显著)
  - serine_ratio_bin:  RF2 HR=2.444 (强风险)  → Final HR=0.906 [0.51-1.62] (不显著)

策略转变:
  v2.4: 首残基芳香族 = 硬约束 (因为RF2保护性)
  v3.0: 首残基芳香族 = 软偏好 (因为Final阶段反转), 权重由阶段权重平衡

作者: CSC-O Team
版本: v3.0
日期: 2026-06-04
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy

from csco_config import AMINO_ACIDS, AROMATIC, HYDROPHOBIC, extract_cdr3_features


# === 默认阶段权重 ===
DEFAULT_FUNNEL_WEIGHTS = {
    'rf2': 0.4,    # RF2通过率权重
    'final': 0.6,  # Final候选率权重 (更高,因为这才是真正目标)
}

# === 阶段感知约束定义 ===
DEFAULT_STAGE_AWARE_CONSTRAINTS = {
    'first_is_aromatic': {
        'rf2_weight': 2.0,       # RF2阶段强正权重(芳香族保护)
        'final_weight': -1.5,    # Final阶段负权重(芳香族风险反转)
        'hard_constraint': False, # 从硬约束降级为软偏好
        'note': 'RF2保护(HR=0.069)但Final反转(HR=3.676, p=1e-9), 设为软偏好',
        'evidence': {
            'rf2_hr': 0.069,
            'rf2_hr_ci': [0.059, 0.080],
            'final_hr': 3.676,
            'final_hr_ci': [2.421, 5.582],
            'final_p_value': 1.0e-9,
        }
    },
    'cdr3_length': {
        'rf2_optimal': [6, 7],
        'final_optimal': [6],    # 长度6在Final阶段可能更优
        'rf2_weight': 1.0,
        'final_weight': 0.5,
        'hard_constraint': True,  # 长度仍为硬约束
        'note': 'RF2保护(HR=0.096)但Final反转(HR=1.928, p=0.007), 长度6可能更优',
        'evidence': {
            'rf2_hr': 0.096,
            'rf2_hr_ci': [0.088, 0.106],
            'final_hr': 1.928,
            'final_hr_ci': [1.198, 3.105],
            'final_p_value': 0.007,
        }
    },
    'glycine_ratio': {
        'rf2_weight': -1.5,      # RF2阶段负权重(甘氨酸风险)
        'final_weight': 0.3,     # Final阶段微弱正权重(不显著)
        'hard_constraint': False,
        'note': 'RF2风险(HR=1.390), Final不显著(HR=0.863, p=0.546), 仍应限制',
        'evidence': {
            'rf2_hr': 1.390,
            'rf2_hr_ci': [1.254, 1.541],
            'final_hr': 0.863,
            'final_hr_ci': [0.535, 1.392],
            'final_p_value': 0.546,
        }
    },
    'serine_ratio': {
        'rf2_weight': -2.0,      # RF2阶段强负权重(丝氨酸强风险)
        'final_weight': 0.2,     # Final阶段微弱正权重(不显著)
        'hard_constraint': False,
        'note': 'RF2强风险(HR=2.444), Final不显著(HR=0.906, p=0.738), 仍应限制',
        'evidence': {
            'rf2_hr': 2.444,
            'rf2_hr_ci': [2.217, 2.695],
            'final_hr': 0.906,
            'final_hr_ci': [0.507, 1.618],
            'final_p_value': 0.738,
        }
    },
}


@dataclass
class FunnelScoreResult:
    """漏斗感知评分结果"""
    sequence: str
    rf2_score: float       # RF2阶段得分
    final_score: float     # Final阶段得分
    combined_score: float  # 加权组合得分
    stage_deltas: Dict[str, float] = field(default_factory=dict)
    constraint_violations: List[str] = field(default_factory=list)
    recommendation: str = ""


class FunnelAwareStrategy:
    """
    漏斗感知策略
    
    核心思想:
    1. 某些约束从"硬约束"降级为"软偏好"(如首残基芳香性)
    2. 用阶段权重计算综合得分, 而非单一阶段最优
    3. 生成策略考虑RF2和Final两阶段的冲突信号
    """
    
    def __init__(
        self,
        stage_aware_constraints: Dict = None,
        funnel_weights: Dict = None,
        base_strategy: Dict = None,
        verbose: bool = True,
    ):
        self.constraints = stage_aware_constraints or DEFAULT_STAGE_AWARE_CONSTRAINTS
        self.funnel_weights = funnel_weights or DEFAULT_FUNNEL_WEIGHTS
        self.base_strategy = base_strategy
        self.verbose = verbose
        
        if self.verbose:
            print(f"[FunnelAwareStrategy] 初始化")
            print(f"  阶段权重: RF2={self.funnel_weights['rf2']}, Final={self.funnel_weights['final']}")
            print(f"  约束数: {len(self.constraints)}")
    
    def generate_v3_strategy(self, base_strategy_path: str = None) -> Dict:
        """
        基于阶段感知约束生成v3.0策略JSON
        
        Args:
            base_strategy_path: v2.4策略文件路径(可选)
            
        Returns:
            v3.0策略字典
        """
        # 加载基础策略
        if base_strategy_path:
            with open(base_strategy_path) as f:
                base = json.load(f)
        elif self.base_strategy:
            base = copy.deepcopy(self.base_strategy)
        else:
            # 使用最小默认策略
            base = {
                'strategy_name': 'CSC-O_v3.0_funnel_aware',
                'version': '3.0',
                'hard_constraints': {
                    'cdr3_length_allowed': [6, 7],
                    'cdr3_length_preferred': [6, 7],
                    'cdr3_last_residue_whitelist': ['Y', 'A', 'H', 'N', 'D', 'W', 'S', 'T', 'V', 'F'],
                    'cdr3_min_positive_count': 0,
                    'cdr3_min_aromatic_first': False,  # 关键变更: 从True降级为False
                },
                'soft_preferences': {
                    'aromatic_min_ratio': 0.08,
                    'glycine_max_ratio': 0.15,
                    'serine_max_ratio': 0.25,
                    'proline_max_count': 1,
                    'hydrophobic_min_ratio': 0.30,
                },
                'anti_patterns': ['GGG', 'SSS', 'LL'],
            }
        
        # === 关键修改1: 首残基芳香族从硬约束降级为软偏好 ===
        if 'hard_constraints' in base:
            hc = base['hard_constraints']
            
            # 移除首残基芳香族硬约束
            hc['cdr3_min_aromatic_first'] = False
            
            # 扩展首残基白名单: 从仅[F,W,Y]扩展到包含非芳香族高潜力残基
            # 基于Final阶段分析: 非芳香族首残基在Final阶段可能更优
            current_whitelist = hc.get('cdr3_first_residue_whitelist', ['F', 'W', 'Y'])
            if set(current_whitelist) <= {'F', 'W', 'Y'}:
                # 当前白名单仅含芳香族, 扩展
                # 添加在Final阶段可能表现好的残基: V(疏水), A(小), D(酸性), T(极性)
                hc['cdr3_first_residue_whitelist'] = ['F', 'W', 'Y', 'V', 'A', 'D', 'T']
                if self.verbose:
                    print(f"  [策略变更] 首残基白名单: {current_whitelist} → {hc['cdr3_first_residue_whitelist']}")
        
        # === 关键修改2: 添加阶段感知约束 ===
        base['stage_aware_constraints'] = {}
        for name, constraint in self.constraints.items():
            base['stage_aware_constraints'][name] = {
                'rf2_weight': constraint['rf2_weight'],
                'final_weight': constraint['final_weight'],
                'hard_constraint': constraint['hard_constraint'],
                'note': constraint.get('note', ''),
            }
        
        # === 关键修改3: 添加阶段权重 ===
        base['funnel_stage_weights'] = self.funnel_weights
        
        # === 关键修改4: 添加首残基软偏好(替代硬约束) ===
        if 'soft_preferences' not in base:
            base['soft_preferences'] = {}
        base['soft_preferences']['first_aromatic_bonus'] = 0.5  # 软偏好加分
        base['soft_preferences']['first_non_aromatic_final_bonus'] = 0.3  # 非芳香族Final阶段加分
        
        # === 关键修改5: 长度6权重提升(因为cdr3_length_bin Final HR反转) ===
        if 'length_generation_weights' in base:
            # 提升长度6的权重
            old_6 = base['length_generation_weights'].get('6', 0.5)
            base['length_generation_weights']['6'] = min(old_6 * 1.3, 0.8)
            if self.verbose:
                print(f"  [策略变更] 长度6生成权重: {old_6:.3f} → {base['length_generation_weights']['6']:.3f}")
        
        # 更新版本信息
        base['strategy_name'] = 'CSC-O_v3.0_funnel_aware'
        base['version'] = '3.0'
        base['description'] = 'Funnel-aware: 首残基芳香族降级为软偏好, 阶段权重平衡RF2/Final'
        
        return base
    
    def score_sequence_funnel(self, cdr3: str) -> FunnelScoreResult:
        """
        用漏斗感知方法评分单条序列
        
        分别计算RF2阶段得分和Final阶段得分, 然后加权组合
        """
        features = extract_cdr3_features(cdr3)
        
        rf2_score = 0.0
        final_score = 0.0
        stage_deltas = {}
        
        # === first_is_aromatic ===
        if features['first_is_aromatic']:
            rf2_score += self.constraints['first_is_aromatic']['rf2_weight']
            final_score += self.constraints['first_is_aromatic']['final_weight']
            stage_deltas['first_aromatic_rf2'] = self.constraints['first_is_aromatic']['rf2_weight']
            stage_deltas['first_aromatic_final'] = self.constraints['first_is_aromatic']['final_weight']
        
        # === cdr3_length ===
        length_constraint = self.constraints['cdr3_length']
        if features['cdr3_len'] in length_constraint['rf2_optimal']:
            rf2_score += length_constraint['rf2_weight']
        if features['cdr3_len'] in length_constraint.get('final_optimal', [6]):
            final_score += length_constraint['final_weight']
        stage_deltas['length_rf2'] = length_constraint['rf2_weight'] if features['cdr3_len'] in length_constraint['rf2_optimal'] else 0
        stage_deltas['length_final'] = length_constraint['final_weight'] if features['cdr3_len'] in length_constraint.get('final_optimal', [6]) else 0
        
        # === glycine_ratio ===
        gly_constraint = self.constraints['glycine_ratio']
        if features['glycine_ratio'] <= 0.20:
            rf2_score += abs(gly_constraint['rf2_weight'])  # 低甘氨酸对RF2有利
            stage_deltas['glycine_rf2'] = abs(gly_constraint['rf2_weight'])
        else:
            rf2_score -= abs(gly_constraint['rf2_weight'])  # 高甘氨酸对RF2不利
            stage_deltas['glycine_rf2'] = -abs(gly_constraint['rf2_weight'])
        
        if features['glycine_ratio'] <= 0.15:
            final_score += gly_constraint['final_weight']
            stage_deltas['glycine_final'] = gly_constraint['final_weight']
        
        # === serine_ratio ===
        ser_constraint = self.constraints['serine_ratio']
        if features['serine_ratio'] <= 0.15:
            rf2_score += abs(ser_constraint['rf2_weight'])  # 低丝氨酸对RF2有利
            stage_deltas['serine_rf2'] = abs(ser_constraint['rf2_weight'])
        else:
            rf2_score -= abs(ser_constraint['rf2_weight'])
            stage_deltas['serine_rf2'] = -abs(ser_constraint['rf2_weight'])
        
        if features['serine_ratio'] <= 0.25:
            final_score += ser_constraint['final_weight']
            stage_deltas['serine_final'] = ser_constraint['final_weight']
        
        # === 加权组合 ===
        w_rf2 = self.funnel_weights['rf2']
        w_final = self.funnel_weights['final']
        combined_score = w_rf2 * rf2_score + w_final * final_score
        
        # === 约束违反检查 ===
        violations = []
        if features['cdr3_len'] not in [6, 7]:
            violations.append('length_not_allowed')
        if features['glycine_ratio'] > 0.20:
            violations.append('glycine_high')
        if features['serine_ratio'] > 0.25:
            violations.append('serine_high')
        
        # === 推荐 ===
        recommendation = self._generate_recommendation(features, rf2_score, final_score)
        
        return FunnelScoreResult(
            sequence=cdr3,
            rf2_score=rf2_score,
            final_score=final_score,
            combined_score=combined_score,
            stage_deltas=stage_deltas,
            constraint_violations=violations,
            recommendation=recommendation,
        )
    
    def _generate_recommendation(self, features: Dict, rf2_score: float, final_score: float) -> str:
        """生成序列改进推荐"""
        recs = []
        
        # 首残基推荐
        if features['first_is_aromatic']:
            if final_score < 0:
                recs.append("首残基芳香族: RF2有利但Final不利, 考虑替换为V/A/D/T")
        else:
            if rf2_score < 0:
                recs.append("首残基非芳香族: Final有利但RF2不利, 考虑替换为F/W/Y")
        
        # 长度推荐
        if features['cdr3_len'] == 7:
            recs.append("长度7: Final阶段HR=1.928(风险), 考虑缩短到6")
        
        # 甘氨酸推荐
        if features['glycine_ratio'] > 0.15:
            recs.append(f"甘氨酸比例{features['glycine_ratio']:.0%}: RF2风险(HR=1.39), 建议降低到≤15%")
        
        # 丝氨酸推荐
        if features['serine_ratio'] > 0.15:
            recs.append(f"丝氨酸比例{features['serine_ratio']:.0%}: RF2强风险(HR=2.44), 建议降低到≤15%")
        
        return "; ".join(recs) if recs else "序列符合漏斗感知约束"
    
    def compare_strategies(
        self,
        sequences: List[str],
        v2_strategy: Dict = None,
    ) -> pd.DataFrame:
        """
        对比v2.4(硬约束)和v3.0(漏斗感知)策略对同一批序列的评分
        
        Args:
            sequences: 待评分序列列表
            v2_strategy: v2.4策略(可选)
            
        Returns:
            对比结果DataFrame
        """
        rows = []
        
        for seq in sequences:
            # v3.0漏斗感知评分
            v3_result = self.score_sequence_funnel(seq)
            
            # v2.4评分(简单: 首残基芳香族=2分, 否则=0)
            v2_score = 2.0 if seq[0] in AROMATIC else 0.0
            v2_pass = seq[0] in AROMATIC  # v2.4硬约束
            
            # 判断v2.4和v3.0策略差异
            v3_prefers = v3_result.combined_score > 0
            strategy_divergence = (v2_pass and not v3_prefers) or (not v2_pass and v3_prefers)
            
            rows.append({
                'sequence': seq,
                'length': len(seq),
                'first_aa': seq[0],
                'first_is_aromatic': seq[0] in AROMATIC,
                'v2_score': v2_score,
                'v2_pass': v2_pass,
                'v3_rf2_score': v3_result.rf2_score,
                'v3_final_score': v3_result.final_score,
                'v3_combined_score': v3_result.combined_score,
                'v3_prefers': v3_prefers,
                'strategy_divergence': strategy_divergence,
                'recommendation': v3_result.recommendation,
            })
        
        return pd.DataFrame(rows)
    
    def save_v3_strategy(self, output_path: str, base_strategy_path: str = None):
        """保存v3.0策略到JSON文件"""
        strategy = self.generate_v3_strategy(base_strategy_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"[FunnelAwareStrategy] v3.0策略已保存到: {output_path}")
        
        return strategy


def run_strategy_comparison(
    data_path: str,
    output_dir: str,
    base_strategy_path: str = None,
    n_sequences: int = 5000,
):
    """
    运行v2.4 vs v3.0策略对比
    
    Args:
        data_path: 原始数据CSV路径
        output_dir: 输出目录
        base_strategy_path: v2.4策略文件路径
        n_sequences: 采样序列数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 获取CDR3序列
    if 'cdr3_sequence' in df.columns:
        sequences = df['cdr3_sequence'].dropna().tolist()[:n_sequences]
    elif 'cdr3' in df.columns:
        sequences = df['cdr3'].dropna().tolist()[:n_sequences]
    else:
        print("错误: 找不到CDR3序列列")
        return
    
    # 初始化策略
    strategy = FunnelAwareStrategy(verbose=True)
    
    # 对比评分
    print(f"\n=== v2.4 vs v3.0 策略对比 (n={len(sequences)}) ===")
    comparison_df = strategy.compare_strategies(sequences)
    
    # 统计
    n_divergence = comparison_df['strategy_divergence'].sum()
    n_aromatic = comparison_df['first_is_aromatic'].sum()
    n_non_aromatic_v3_preferred = ((~comparison_df['first_is_aromatic']) & (comparison_df['v3_prefers'])).sum()
    
    print(f"\n策略分歧序列数: {n_divergence}/{len(sequences)} ({n_divergence/len(sequences)*100:.1f}%)")
    print(f"芳香族首残基: {n_aromatic} ({n_aromatic/len(sequences)*100:.1f}%)")
    print(f"非芳香族但v3.0偏好: {n_non_aromatic_v3_preferred}")
    
    # 保存结果
    comparison_df.to_csv(output_dir / 'strategy_comparison_v2_vs_v3.csv', index=False)
    
    # 生成v3.0策略
    v3_strategy = strategy.save_v3_strategy(
        str(output_dir / 'design_strategy_v3.0.json'),
        base_strategy_path=base_strategy_path,
    )
    
    # 对FC正样本单独分析
    if 'final_candidate' in df.columns:
        fc_df = df[df['final_candidate'] == True]
        if len(fc_df) > 0 and 'cdr3_sequence' in fc_df.columns:
            fc_sequences = fc_df['cdr3_sequence'].dropna().tolist()
            fc_comparison = strategy.compare_strategies(fc_sequences)
            fc_comparison.to_csv(output_dir / 'fc_positive_strategy_comparison.csv', index=False)
            
            fc_aromatic_rate = fc_comparison['first_is_aromatic'].mean()
            fc_v3_preferred = fc_comparison['v3_prefers'].mean()
            print(f"\nFC正样本 (n={len(fc_sequences)}):")
            print(f"  芳香族首残基率: {fc_aromatic_rate:.1%}")
            print(f"  v3.0偏好率: {fc_v3_preferred:.1%}")
    
    return comparison_df, v3_strategy


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CSC-O Funnel-Aware Strategy')
    parser.add_argument('--data', required=True, help='原始数据CSV路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--base-strategy', default=None, help='v2.4策略文件路径')
    parser.add_argument('--n-sequences', type=int, default=5000, help='采样序列数')
    
    args = parser.parse_args()
    
    run_strategy_comparison(
        data_path=args.data,
        output_dir=args.output,
        base_strategy_path=args.base_strategy,
        n_sequences=args.n_sequences,
    )
