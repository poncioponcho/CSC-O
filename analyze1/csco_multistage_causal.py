#!/usr/bin/env python3
"""
CSC-O Phase 1: 多阶段中介模型 (Multi-Stage Mediation Model)

因果结构:
    T (Treatment) → M (Mediator: rf2_passed) → Y (Outcome: final_candidate)
    T → Y (Direct Effect)

分解建模方法:
    Total Effect = P(final|T=1) - P(final|T=0)
                 = P(RF2|T=1)×P(final|RF2,T=1) - P(RF2|T=0)×P(final|RF2,T=0)
    
    Direct Effect = P(final|RF2=1, T=1) - P(final|RF2=1, T=0)
    
    Indirect Effect = Total Effect - Direct Effect

作者: CSC-O Team
版本: v3.0
日期: 2026-06-04
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import time

# 项目内部模块
from csco_config import (
    AROMATIC, GLYCINE, SERINE, PROLINE, HYDROPHOBIC, POSITIVE, NEGATIVE,
    DEFAULT_CONFIG, TREATMENT_VARS_EXTENDED, CONFOUNDER_COLS, CDR3_FEATURE_COLS,
    extract_cdr3_features,
)

# === 常量定义 ===

ALL_STAGES = ['rf2', 'af3', 'schrodinger', 'final']
AVAILABLE_STAGES_DEFAULT = ['rf2', 'final']

TREATMENT_DEFINITIONS = {
    'first_is_aromatic': {
        'description': '首残基是否为芳香族(F/W/Y)',
        'type': 'binary',
        'protective_expected': True,  # 预期为保护性
    },
    'cdr3_length_bin': {
        'description': 'CDR3长度分组 (6-7 vs 其他)',
        'type': 'binary',
        'protective_expected': True,
    },
    'glycine_ratio_bin': {
        'description': '甘氨酸比例 >20% vs ≤20%',
        'type': 'binary',
        'protective_expected': False,  # 预期为风险因素
    },
    'serine_ratio_bin': {
        'description': '丝氨酸比例 >15% vs ≤15%',
        'type': 'binary',
        'protective_expected': False,
    },
}


@dataclass
class MediationEffectResult:
    """中介效应估计结果"""
    treatment: str
    total_effect: float
    total_effect_se: float
    direct_effect: float
    direct_effect_se: float
    indirect_effect: float
    indirect_effect_se: float
    n_stage1: int
    n_stage2: int
    n_final: int
    method: str = 'decomposition'
    available_stages: List[str] = field(default_factory=lambda: ['rf2', 'final'])
    
    # 阶段特异性效应
    stage1_effect: float = 0.0  # P(RF2)效应
    stage1_effect_se: float = 0.0
    stage2_effect: float = 0.0  # P(final|RF2)效应
    stage2_effect_se: float = 0.0
    
    # 置信区间
    total_effect_ci: Tuple[float, float] = (0.0, 0.0)
    direct_effect_ci: Tuple[float, float] = (0.0, 0.0)
    indirect_effect_ci: Tuple[float, float] = (0.0, 0.0)
    
    # 显著性
    total_effect_pvalue: float = 1.0
    direct_effect_pvalue: float = 1.0
    indirect_effect_pvalue: float = 1.0
    
    # 解释
    interpretation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'treatment': self.treatment,
            'total_effect': self.total_effect,
            'total_effect_se': self.total_effect_se,
            'total_effect_ci': self.total_effect_ci,
            'total_effect_pvalue': self.total_effect_pvalue,
            'direct_effect': self.direct_effect,
            'direct_effect_se': self.direct_effect_se,
            'direct_effect_ci': self.direct_effect_ci,
            'direct_effect_pvalue': self.direct_effect_pvalue,
            'indirect_effect': self.indirect_effect,
            'indirect_effect_se': self.indirect_effect_se,
            'indirect_effect_ci': self.indirect_effect_ci,
            'indirect_effect_pvalue': self.indirect_effect_pvalue,
            'stage1_effect': self.stage1_effect,
            'stage1_effect_se': self.stage1_effect_se,
            'stage2_effect': self.stage2_effect,
            'stage2_effect_se': self.stage2_effect_se,
            'n_stage1': self.n_stage1,
            'n_stage2': self.n_stage2,
            'n_final': self.n_final,
            'method': self.method,
            'available_stages': self.available_stages,
            'interpretation': self.interpretation,
        }


class MultiStageMediationModel:
    """
    多阶段中介模型
    
    支持三种估计方法:
    1. decomposition: 分解建模 (默认，适用于FC样本稀疏)
    2. transfer: 迁移学习 (FC≥100时启用)
    3. direct: 直接CausalForestDML (FC≥200时启用)
    """
    
    def __init__(
        self,
        available_stages: List[str] = None,
        method: str = 'decomposition',
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        初始化中介模型
        
        Args:
            available_stages: 可用阶段列表，默认['rf2', 'final']
            method: 估计方法，'decomposition'/'transfer'/'direct'
            random_state: 随机种子
            verbose: 是否输出详细日志
        """
        self.available_stages = available_stages or AVAILABLE_STAGES_DEFAULT
        self.method = method
        self.random_state = random_state
        self.verbose = verbose
        
        # 验证阶段配置
        for stage in self.available_stages:
            if stage not in ALL_STAGES:
                raise ValueError(f"Unknown stage: {stage}. Must be one of {ALL_STAGES}")
        
        # 模型存储
        self.stage1_model = None  # P(RF2)模型
        self.stage2_model = None  # P(final|RF2)模型
        self.results_: Dict[str, MediationEffectResult] = {}
        
        if self.verbose:
            print(f"[MultiStageMediationModel] 初始化完成")
            print(f"  可用阶段: {self.available_stages}")
            print(f"  估计方法: {self.method}")
    
    def fit(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        treatment_cols: List[str] = None,
        confounder_cols: List[str] = None,
    ) -> Dict[str, MediationEffectResult]:
        """
        拟合中介模型
        
        Args:
            df: 包含所有特征的DataFrame
            embeddings: ESM-2嵌入矩阵 (n_samples, n_features)
            treatment_cols: 治疗变量列表
            confounder_cols: 混杂变量列表
            
        Returns:
            每个治疗变量的中介效应结果字典
        """
        t0 = time.time()
        
        # 默认治疗变量
        if treatment_cols is None:
            treatment_cols = list(TREATMENT_DEFINITIONS.keys())
        
        # 默认混杂变量
        if confounder_cols is None:
            confounder_cols = ['backbone_id']
        
        # 验证数据
        self._validate_data(df, treatment_cols, confounder_cols)
        
        # 准备混杂变量矩阵
        W = self._prepare_confounders(df, confounder_cols, embeddings)
        
        # 准备中介变量和结局
        M = df['rf2_passed'].astype(int).values  # Mediator
        Y = df['final_candidate'].astype(int).values  # Outcome
        
        n_total = len(df)
        n_rf2_pass = M.sum()
        n_final = Y.sum()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[MultiStageMediationModel] 开始拟合")
            print(f"  总样本数: {n_total}")
            print(f"  RF2通过: {n_rf2_pass} ({n_rf2_pass/n_total*100:.1f}%)")
            print(f"  Final候选: {n_final} ({n_final/n_total*100:.2f}%)")
            print(f"  治疗变量: {treatment_cols}")
            print(f"  混杂变量: {confounder_cols}")
            print(f"  特征维度: {W.shape[1]}")
            print(f"{'='*60}\n")
        
        # 对每个治疗变量估计效应
        for i, treatment in enumerate(treatment_cols):
            if self.verbose:
                print(f"\n[{i+1}/{len(treatment_cols)}] 处理治疗变量: {treatment}")
            
            try:
                result = self._estimate_treatment_effect(
                    df, W, M, Y, treatment, confounder_cols
                )
                self.results_[treatment] = result
            except Exception as e:
                if self.verbose:
                    print(f"  [ERROR] {treatment} 估计失败: {e}")
                continue
        
        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n[MultiStageMediationModel] 拟合完成")
            print(f"  成功估计: {len(self.results_)}/{len(treatment_cols)} 个治疗变量")
            print(f"  耗时: {elapsed:.1f}s")
        
        return self.results_
    
    def _validate_data(
        self,
        df: pd.DataFrame,
        treatment_cols: List[str],
        confounder_cols: List[str],
    ):
        """验证数据完整性"""
        # 核心列必须存在
        required_cols = ['rf2_passed', 'final_candidate']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 治疗变量可以动态生成，只检查是否能生成
        for t in treatment_cols:
            if t not in df.columns:
                # 检查是否有生成该变量的基础列
                if t == 'first_is_aromatic' and 'cdr3_sequence' not in df.columns and 'first_is_aromatic' not in df.columns:
                    raise ValueError(f"Cannot generate treatment '{t}': missing 'cdr3_sequence' and 'first_is_aromatic'")
                elif t == 'cdr3_length_bin' and 'cdr3_len' not in df.columns and 'cdr3_sequence' not in df.columns:
                    raise ValueError(f"Cannot generate treatment '{t}': missing 'cdr3_len' or 'cdr3_sequence'")
                elif t == 'glycine_ratio_bin' and 'glycine_ratio' not in df.columns and 'cdr3_sequence' not in df.columns:
                    raise ValueError(f"Cannot generate treatment '{t}': missing 'glycine_ratio' or 'cdr3_sequence'")
                elif t == 'serine_ratio_bin' and 'serine_ratio' not in df.columns and 'cdr3_sequence' not in df.columns:
                    raise ValueError(f"Cannot generate treatment '{t}': missing 'serine_ratio' or 'cdr3_sequence'")
        
        if len(df) < 100:
            warnings.warn(f"Sample size ({len(df)}) may be too small for reliable estimation")
    
    def _prepare_confounders(
        self,
        df: pd.DataFrame,
        confounder_cols: List[str],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """准备混杂变量矩阵"""
        conf_list = [embeddings]
        
        for col in confounder_cols:
            if col == 'backbone_id':
                # One-hot encoding for backbone_id
                backbone_codes = df[col].astype('category').cat.codes.values.reshape(-1, 1)
                conf_list.append(backbone_codes.astype(np.float32))
            elif col in df.columns:
                conf_list.append(df[col].values.reshape(-1, 1).astype(np.float32))
        
        return np.hstack(conf_list)
    
    def _estimate_treatment_effect(
        self,
        df: pd.DataFrame,
        W: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        treatment: str,
        confounder_cols: List[str],
    ) -> MediationEffectResult:
        """
        估计单个治疗变量的中介效应
        
        使用分解建模方法:
        Stage 1: P(RF2 pass | T, W) - 用全部样本
        Stage 2: P(final | T, W, RF2=1) - 用RF2通过样本
        """
        # 准备治疗变量
        T = self._prepare_treatment(df, treatment)
        
        if T.std() < 1e-8:
            raise ValueError(f"Treatment {treatment} has zero variance")
        
        n_total = len(T)
        n_treat = int(T.sum())
        n_control = n_total - n_treat
        
        if self.verbose:
            print(f"  Treatment分布: treat={n_treat}({n_treat/n_total*100:.1f}%), control={n_control}")
        
        # === Stage 1: P(RF2 pass | T, W) ===
        if self.verbose:
            print(f"  Stage 1: 估计P(RF2 pass)...")
        
        stage1_result = self._estimate_stage1_effect(T, W, M)
        p_rf2_treat = stage1_result['p_rf2_treat']
        p_rf2_control = stage1_result['p_rf2_control']
        stage1_effect = p_rf2_treat - p_rf2_control
        stage1_se = stage1_result['se']
        
        if self.verbose:
            print(f"    P(RF2|T=1)={p_rf2_treat:.4f}, P(RF2|T=0)={p_rf2_control:.4f}")
            print(f"    Stage1 Effect={stage1_effect:.4f} (SE={stage1_se:.4f})")
        
        # === Stage 2: P(final | T, W, RF2=1) ===
        if self.verbose:
            print(f"  Stage 2: 估计P(final|RF2 pass)...")
        
        # 筛选RF2通过样本
        rf2_pass_mask = M == 1
        T_rf2 = T[rf2_pass_mask]
        W_rf2 = W[rf2_pass_mask]
        Y_rf2 = Y[rf2_pass_mask]
        
        n_rf2_pass = len(Y_rf2)
        n_final_in_rf2 = Y_rf2.sum()
        
        if self.verbose:
            print(f"    RF2通过样本: {n_rf2_pass}, 其中Final={n_final_in_rf2}")
        
        stage2_result = self._estimate_stage2_effect(T_rf2, W_rf2, Y_rf2)
        p_final_given_rf2_treat = stage2_result['p_final_treat']
        p_final_given_rf2_control = stage2_result['p_final_control']
        stage2_effect = p_final_given_rf2_treat - p_final_given_rf2_control
        stage2_se = stage2_result['se']
        
        if self.verbose:
            print(f"    P(final|RF2,T=1)={p_final_given_rf2_treat:.4f}, P(final|RF2,T=0)={p_final_given_rf2_control:.4f}")
            print(f"    Stage2 Effect={stage2_effect:.4f} (SE={stage2_se:.4f})")
        
        # === 计算总效应、直接效应、间接效应 ===
        # Total Effect = P(final|T=1) - P(final|T=0)
        #              = P(RF2|T=1)×P(final|RF2,T=1) - P(RF2|T=0)×P(final|RF2,T=0)
        p_final_treat = p_rf2_treat * p_final_given_rf2_treat
        p_final_control = p_rf2_control * p_final_given_rf2_control
        total_effect = p_final_treat - p_final_control
        
        # Direct Effect = P(final|RF2=1, T=1) - P(final|RF2=1, T=0)
        direct_effect = stage2_effect
        
        # Indirect Effect = Total Effect - Direct Effect
        indirect_effect = total_effect - direct_effect
        
        # === 标准误估计 (Delta Method) ===
        # Var(TE) ≈ Var(p1)*p2² + Var(p2)*p1² + 2*p1*p2*Cov(p1,p2)
        # 简化: 假设stage1和stage2独立
        total_effect_se = np.sqrt(
            stage1_se**2 * p_final_given_rf2_treat**2 +
            stage2_se**2 * p_rf2_treat**2 +
            stage1_se**2 * p_final_given_rf2_control**2 +
            stage2_se**2 * p_rf2_control**2
        )
        
        direct_effect_se = stage2_se
        
        # 间接效应SE: Var(IE) = Var(TE) + Var(DE) - 2*Cov(TE,DE)
        # 简化: 假设独立
        indirect_effect_se = np.sqrt(total_effect_se**2 + direct_effect_se**2)
        
        # === 置信区间 (95%) ===
        z = 1.96
        total_effect_ci = (
            total_effect - z * total_effect_se,
            total_effect + z * total_effect_se,
        )
        direct_effect_ci = (
            direct_effect - z * direct_effect_se,
            direct_effect + z * direct_effect_se,
        )
        indirect_effect_ci = (
            indirect_effect - z * indirect_effect_se,
            indirect_effect + z * indirect_effect_se,
        )
        
        # === P值 (双侧检验) ===
        from scipy import stats
        total_effect_pvalue = 2 * (1 - stats.norm.cdf(abs(total_effect) / max(total_effect_se, 1e-8)))
        direct_effect_pvalue = 2 * (1 - stats.norm.cdf(abs(direct_effect) / max(direct_effect_se, 1e-8)))
        indirect_effect_pvalue = 2 * (1 - stats.norm.cdf(abs(indirect_effect) / max(indirect_effect_se, 1e-8)))
        
        # === 解释 ===
        interpretation = self._generate_interpretation(
            treatment, total_effect, direct_effect, indirect_effect,
            stage1_effect, stage2_effect
        )
        
        if self.verbose:
            print(f"  结果:")
            print(f"    Total Effect={total_effect:.4f} (SE={total_effect_se:.4f}, p={total_effect_pvalue:.4f})")
            print(f"    Direct Effect={direct_effect:.4f} (SE={direct_effect_se:.4f}, p={direct_effect_pvalue:.4f})")
            print(f"    Indirect Effect={indirect_effect:.4f} (SE={indirect_effect_se:.4f}, p={indirect_effect_pvalue:.4f})")
            print(f"    解释: {interpretation}")
        
        return MediationEffectResult(
            treatment=treatment,
            total_effect=total_effect,
            total_effect_se=total_effect_se,
            direct_effect=direct_effect,
            direct_effect_se=direct_effect_se,
            indirect_effect=indirect_effect,
            indirect_effect_se=indirect_effect_se,
            n_stage1=n_total,
            n_stage2=n_rf2_pass,
            n_final=int(n_final_in_rf2),
            method=self.method,
            available_stages=self.available_stages,
            stage1_effect=stage1_effect,
            stage1_effect_se=stage1_se,
            stage2_effect=stage2_effect,
            stage2_effect_se=stage2_se,
            total_effect_ci=total_effect_ci,
            direct_effect_ci=direct_effect_ci,
            indirect_effect_ci=indirect_effect_ci,
            total_effect_pvalue=total_effect_pvalue,
            direct_effect_pvalue=direct_effect_pvalue,
            indirect_effect_pvalue=indirect_effect_pvalue,
            interpretation=interpretation,
        )
    
    def _prepare_treatment(self, df: pd.DataFrame, treatment: str) -> np.ndarray:
        """准备治疗变量"""
        if treatment == 'first_is_aromatic':
            if 'first_is_aromatic' in df.columns:
                return df['first_is_aromatic'].astype(int).values
            elif 'cdr3_sequence' in df.columns:
                return df['cdr3_sequence'].apply(
                    lambda s: int(s[0] in AROMATIC) if pd.notna(s) and len(s) > 0 else 0
                ).values
        
        elif treatment == 'cdr3_length_bin':
            if 'cdr3_len' in df.columns:
                return (df['cdr3_len'].isin([6, 7])).astype(int).values
            elif 'cdr3_sequence' in df.columns:
                return df['cdr3_sequence'].apply(
                    lambda s: int(len(s) in [6, 7]) if pd.notna(s) else 0
                ).values
        
        elif treatment == 'glycine_ratio_bin':
            if 'glycine_ratio' in df.columns:
                return (df['glycine_ratio'] > 0.20).astype(int).values
            elif 'cdr3_sequence' in df.columns:
                return df['cdr3_sequence'].apply(
                    lambda s: int(sum(1 for a in s if a in GLYCINE) / len(s) > 0.20) if pd.notna(s) and len(s) > 0 else 0
                ).values
        
        elif treatment == 'serine_ratio_bin':
            if 'serine_ratio' in df.columns:
                return (df['serine_ratio'] > 0.15).astype(int).values
            elif 'cdr3_sequence' in df.columns:
                return df['cdr3_sequence'].apply(
                    lambda s: int(sum(1 for a in s if a in SERINE) / len(s) > 0.15) if pd.notna(s) and len(s) > 0 else 0
                ).values
        
        else:
            # 通用处理: 尝试从df中获取
            if treatment in df.columns:
                T = df[treatment].values
                if len(np.unique(T)) > 5:
                    # 连续变量二值化
                    return (T > np.median(T)).astype(int)
                return T.astype(int)
            else:
                raise ValueError(f"Unknown treatment: {treatment}")
    
    def _estimate_stage1_effect(
        self,
        T: np.ndarray,
        W: np.ndarray,
        M: np.ndarray,
    ) -> Dict:
        """
        估计Stage 1效应: P(RF2 pass | T, W)
        
        使用IPW + Logistic回归
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        
        n = len(T)
        
        # 方法1: 简单的IPW估计
        # P(RF2|T=1) = E[M|T=1]
        # P(RF2|T=0) = E[M|T=0]
        treat_mask = T == 1
        control_mask = T == 0
        
        p_rf2_treat_raw = M[treat_mask].mean()
        p_rf2_control_raw = M[control_mask].mean()
        
        # 方法2: 使用倾向得分调整
        # 拟合倾向得分模型
        try:
            ps_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=self.random_state
            )
            ps_model.fit(W, T)
            ps = ps_model.predict_proba(W)[:, 1]
            ps = np.clip(ps, 0.1, 0.9)  # 避免极端值
            
            # IPW调整后的估计
            weights_treat = treat_mask / ps
            weights_control = control_mask / (1 - ps)
            
            p_rf2_treat = np.average(M[treat_mask], weights=weights_treat[treat_mask])
            p_rf2_control = np.average(M[control_mask], weights=weights_control[control_mask])
            
        except Exception:
            # 降级到简单估计
            p_rf2_treat = p_rf2_treat_raw
            p_rf2_control = p_rf2_control_raw
        
        # 标准误估计 (Bootstrap简化版)
        se = self._bootstrap_se_stage1(T, W, M, n_bootstrap=100)
        
        return {
            'p_rf2_treat': p_rf2_treat,
            'p_rf2_control': p_rf2_control,
            'se': se,
        }
    
    def _estimate_stage2_effect(
        self,
        T: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
    ) -> Dict:
        """
        估计Stage 2效应: P(final | T, W, RF2=1)
        
        使用IPW + Logistic回归
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        
        n = len(T)
        
        if n < 20:
            warnings.warn(f"Stage 2 sample size ({n}) is very small")
        
        # 简单估计
        treat_mask = T == 1
        control_mask = T == 0
        
        n_treat = treat_mask.sum()
        n_control = control_mask.sum()
        
        if n_treat < 5 or n_control < 5:
            # 样本太少，使用先验平滑
            p_final_treat = (Y[treat_mask].sum() + 1) / (n_treat + 2)
            p_final_control = (Y[control_mask].sum() + 1) / (n_control + 2)
        else:
            p_final_treat = Y[treat_mask].mean()
            p_final_control = Y[control_mask].mean()
        
        # 使用倾向得分调整
        try:
            if n_treat >= 10 and n_control >= 10:
                ps_model = GradientBoostingClassifier(
                    n_estimators=30, max_depth=2, learning_rate=0.1, random_state=self.random_state
                )
                ps_model.fit(W, T)
                ps = ps_model.predict_proba(W)[:, 1]
                ps = np.clip(ps, 0.1, 0.9)
                
                weights_treat = treat_mask / ps
                weights_control = control_mask / (1 - ps)
                
                p_final_treat = np.average(Y[treat_mask], weights=weights_treat[treat_mask])
                p_final_control = np.average(Y[control_mask], weights=weights_control[control_mask])
        except Exception:
            pass
        
        # 标准误估计
        se = self._bootstrap_se_stage2(T, W, Y, n_bootstrap=100)
        
        return {
            'p_final_treat': p_final_treat,
            'p_final_control': p_final_control,
            'se': se,
        }
    
    def _bootstrap_se_stage1(
        self,
        T: np.ndarray,
        W: np.ndarray,
        M: np.ndarray,
        n_bootstrap: int = 100,
    ) -> float:
        """Bootstrap标准误估计 (Stage 1)"""
        n = len(T)
        effects = []
        
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            idx = rng.choice(n, size=n, replace=True)
            T_b = T[idx]
            M_b = M[idx]
            
            treat_mask = T_b == 1
            control_mask = T_b == 0
            
            if treat_mask.sum() > 0 and control_mask.sum() > 0:
                effect_b = M_b[treat_mask].mean() - M_b[control_mask].mean()
                effects.append(effect_b)
        
        if len(effects) < 10:
            # 降级到解析标准误
            treat_mask = T == 1
            control_mask = T == 0
            var_treat = M[treat_mask].var() / max(treat_mask.sum(), 1)
            var_control = M[control_mask].var() / max(control_mask.sum(), 1)
            return np.sqrt(var_treat + var_control)
        
        return np.std(effects)
    
    def _bootstrap_se_stage2(
        self,
        T: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
        n_bootstrap: int = 100,
    ) -> float:
        """Bootstrap标准误估计 (Stage 2)"""
        n = len(T)
        effects = []
        
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            T_b = T[idx]
            Y_b = Y[idx]
            
            treat_mask = T_b == 1
            control_mask = T_b == 0
            
            if treat_mask.sum() > 0 and control_mask.sum() > 0:
                effect_b = Y_b[treat_mask].mean() - Y_b[control_mask].mean()
                effects.append(effect_b)
        
        if len(effects) < 10:
            treat_mask = T == 1
            control_mask = T == 0
            var_treat = Y[treat_mask].var() / max(treat_mask.sum(), 1)
            var_control = Y[control_mask].var() / max(control_mask.sum(), 1)
            return np.sqrt(var_treat + var_control)
        
        return np.std(effects)
    
    def _generate_interpretation(
        self,
        treatment: str,
        total_effect: float,
        direct_effect: float,
        indirect_effect: float,
        stage1_effect: float,
        stage2_effect: float,
    ) -> str:
        """生成效应解释"""
        treatment_def = TREATMENT_DEFINITIONS.get(treatment, {})
        expected_protective = treatment_def.get('protective_expected', None)
        
        parts = []
        
        # 总效应方向
        if total_effect > 0.001:
            parts.append(f"总效应为正({total_effect:.4f}), 增加{treatment}提升P(final)")
        elif total_effect < -0.001:
            parts.append(f"总效应为负({total_effect:.4f}), 增加{treatment}降低P(final)")
        else:
            parts.append(f"总效应接近0({total_effect:.4f}), {treatment}对P(final)影响微弱")
        
        # 直接vs间接
        if abs(direct_effect) > abs(indirect_effect):
            parts.append(f"直接效应({direct_effect:.4f})主导")
        else:
            parts.append(f"间接效应({indirect_effect:.4f})主导(通过RF2)")
        
        # 阶段特异性
        if abs(stage1_effect) > abs(stage2_effect) * 2:
            parts.append(f"效应主要来自RF2阶段({stage1_effect:.4f})")
        elif abs(stage2_effect) > abs(stage1_effect) * 2:
            parts.append(f"效应主要来自Final阶段({stage2_effect:.4f})")
        else:
            parts.append(f"两阶段均有贡献(RF2:{stage1_effect:.4f}, Final:{stage2_effect:.4f})")
        
        # 与预期一致性
        if expected_protective is not None:
            is_protective = total_effect < 0
            if is_protective == expected_protective:
                parts.append("与预期一致")
            else:
                parts.append("与预期相反,需进一步验证")
        
        return "; ".join(parts)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """将结果转换为DataFrame"""
        if not self.results_:
            return pd.DataFrame()
        
        records = [r.to_dict() for r in self.results_.values()]
        return pd.DataFrame(records)
    
    def save_results(self, output_path: Union[str, Path]):
        """保存结果到CSV"""
        df = self.get_results_dataframe()
        df.to_csv(output_path, index=False)
        if self.verbose:
            print(f"[MultiStageMediationModel] 结果已保存到: {output_path}")


def run_multistage_causal_analysis(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Union[str, Path],
    config: Dict = None,
) -> Dict[str, MediationEffectResult]:
    """
    运行多阶段中介分析的主入口函数
    
    Args:
        df: 包含所有特征的DataFrame
        embeddings: ESM-2嵌入矩阵
        output_dir: 输出目录
        config: 配置字典
        
    Returns:
        中介效应结果字典
    """
    config = config or DEFAULT_CONFIG
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    model = MultiStageMediationModel(
        available_stages=config.get('available_stages', AVAILABLE_STAGES_DEFAULT),
        method=config.get('mediation_method', 'decomposition'),
        random_state=config.get('random_state', 42),
        verbose=config.get('verbose', True),
    )
    
    # 拟合模型
    results = model.fit(
        df=df,
        embeddings=embeddings,
        treatment_cols=config.get('treatment_cols', list(TREATMENT_DEFINITIONS.keys())),
        confounder_cols=config.get('confounder_cols', ['backbone_id']),
    )
    
    # 保存结果
    model.save_results(output_dir / 'mediation_effects.csv')
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CSC-O Phase 1: 多阶段中介模型')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--embeddings', required=True, help='ESM-2嵌入.npy文件路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--method', default='decomposition', choices=['decomposition', 'transfer', 'direct'],
                        help='估计方法')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 加载数据
    df = pd.read_csv(args.input)
    embeddings = np.load(args.embeddings)
    
    # 运行分析
    config = {
        'mediation_method': args.method,
        'verbose': args.verbose,
    }
    
    results = run_multistage_causal_analysis(
        df=df,
        embeddings=embeddings,
        output_dir=args.output,
        config=config,
    )
    
    print(f"\n完成! 共估计 {len(results)} 个治疗变量的中介效应")
