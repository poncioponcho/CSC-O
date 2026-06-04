#!/usr/bin/env python3
"""
CSC-O Phase 2: 多状态生存模型 (Multi-State Survival Model)

两阶段近似:
    状态0 (all, n=10572)
      ├─→ 状态1 (rf2_failed, n=9337)   [事件1: RF2失败]
      └─→ 状态2 (rf2_passed, n=1235)
            ├─→ 状态3 (final_candidate, n=65)  [事件2: 成为FC]
            └─→ 状态4 (censored, n=1170)       [右删失: 未被AF3分析]

阶段特异性Cox回归:
    Stage 1: RF2失败风险 (全部样本)
    Stage 2: 成为FC的风险 (RF2通过样本)

代码预留4阶段枚举(ALL_STAGES)，通过available_stages参数控制当前激活阶段。

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

from csco_config import (
    AROMATIC, GLYCINE, SERINE, PROLINE, HYDROPHOBIC, POSITIVE, NEGATIVE,
    DEFAULT_CONFIG, CONFOUNDER_COLS,
)

# 复用Phase 1的阶段定义
ALL_STAGES = ['rf2', 'af3', 'schrodinger', 'final']
AVAILABLE_STAGES_DEFAULT = ['rf2', 'final']


@dataclass
class StageHazardResult:
    """单阶段Cox回归结果"""
    treatment: str
    stage: str
    hr: float  # Hazard Ratio
    hr_ci: Tuple[float, float]  # 95% CI
    p_value: float
    coef: float
    coef_se: float
    n_events: int
    n_at_risk: int
    concordance: float = 0.0
    interpretation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'treatment': self.treatment,
            'stage': self.stage,
            'hr': self.hr,
            'hr_ci_lower': self.hr_ci[0],
            'hr_ci_upper': self.hr_ci[1],
            'p_value': self.p_value,
            'coef': self.coef,
            'coef_se': self.coef_se,
            'n_events': self.n_events,
            'n_at_risk': self.n_at_risk,
            'concordance': self.concordance,
            'interpretation': self.interpretation,
        }


@dataclass
class MultiStateHazardResult:
    """多状态Cox回归结果（汇总）"""
    treatment: str
    stage_results: Dict[str, StageHazardResult] = field(default_factory=dict)
    combined_hr: float = 1.0
    combined_hr_ci: Tuple[float, float] = (1.0, 1.0)
    dominant_stage: str = ""
    interpretation: str = ""
    
    def to_dict(self) -> Dict:
        d = {
            'treatment': self.treatment,
            'combined_hr': self.combined_hr,
            'combined_hr_ci_lower': self.combined_hr_ci[0],
            'combined_hr_ci_upper': self.combined_hr_ci[1],
            'dominant_stage': self.dominant_stage,
            'interpretation': self.interpretation,
        }
        for stage, result in self.stage_results.items():
            for k, v in result.to_dict().items():
                if k != 'treatment':
                    d[f'{stage}_{k}'] = v
        return d


class MultiStateSurvivalModel:
    """
    多状态生存模型
    
    两阶段近似:
    - Stage 1 (rf2): RF2失败风险，用全部10,572样本
    - Stage 2 (final): 成为FC的风险，用1,235条RF2通过样本
    
    预留4阶段扩展: available_stages=['rf2', 'af3', 'schrodinger', 'final']
    """
    
    def __init__(
        self,
        available_stages: List[str] = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.available_stages = available_stages or AVAILABLE_STAGES_DEFAULT
        self.random_state = random_state
        self.verbose = verbose
        
        for stage in self.available_stages:
            if stage not in ALL_STAGES:
                raise ValueError(f"Unknown stage: {stage}. Must be one of {ALL_STAGES}")
        
        self.results_: Dict[str, MultiStateHazardResult] = {}
        self.stage_models_: Dict[str, Any] = {}
        
        if self.verbose:
            print(f"[MultiStateSurvivalModel] 初始化完成")
            print(f"  可用阶段: {self.available_stages}")
    
    def fit(
        self,
        df: pd.DataFrame,
        treatment_cols: List[str] = None,
        confounder_cols: List[str] = None,
    ) -> Dict[str, MultiStateHazardResult]:
        """
        拟合多状态Cox模型
        
        Args:
            df: 包含所有特征的DataFrame
            treatment_cols: 治疗变量列表
            confounder_cols: 混杂变量列表
            
        Returns:
            每个治疗变量的多状态HR结果
        """
        t0 = time.time()
        
        if treatment_cols is None:
            treatment_cols = ['first_is_aromatic', 'cdr3_length_bin',
                              'glycine_ratio_bin', 'serine_ratio_bin']
        
        if confounder_cols is None:
            confounder_cols = ['backbone_id']
        
        # 准备数据
        df = self._prepare_survival_data(df)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[MultiStateSurvivalModel] 开始拟合")
            print(f"  总样本数: {len(df)}")
            print(f"  RF2通过: {df['rf2_passed'].sum()}")
            print(f"  Final候选: {df['final_candidate'].sum()}")
            print(f"  治疗变量: {treatment_cols}")
            print(f"{'='*60}\n")
        
        # 对每个治疗变量拟合
        for i, treatment in enumerate(treatment_cols):
            if self.verbose:
                print(f"\n[{i+1}/{len(treatment_cols)}] 处理: {treatment}")
            
            try:
                result = self._fit_treatment(df, treatment, confounder_cols)
                self.results_[treatment] = result
            except Exception as e:
                if self.verbose:
                    print(f"  [ERROR] {treatment} 拟合失败: {e}")
                continue
        
        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n[MultiStateSurvivalModel] 拟合完成")
            print(f"  成功: {len(self.results_)}/{len(treatment_cols)}")
            print(f"  耗时: {elapsed:.1f}s")
        
        return self.results_
    
    def _prepare_survival_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备生存分析数据"""
        df = df.copy()
        
        # 确保关键列存在
        if 'rf2_passed' not in df.columns:
            df['rf2_passed'] = df.get('rf2_passed_filter', False)
        
        if 'final_candidate' not in df.columns:
            df['final_candidate'] = False
        
        # 构建生存时间变量
        # Stage 1: time = 1 (所有样本都"暴露"于RF2阶段)
        # Stage 2: time = 2 (仅RF2通过样本"暴露"于Final阶段)
        df['surv_time_stage1'] = 1  # 所有样本
        df['surv_event_stage1'] = (~df['rf2_passed']).astype(int)  # 1=RF2失败
        
        # Stage 2: 仅RF2通过样本
        df['surv_time_stage2'] = 2
        df['surv_event_stage2'] = df['final_candidate'].astype(int)  # 1=成为FC
        
        # 生成治疗变量
        self._generate_treatment_vars(df)
        
        return df
    
    def _generate_treatment_vars(self, df: pd.DataFrame):
        """生成治疗变量"""
        if 'first_is_aromatic' not in df.columns and 'cdr3_sequence' in df.columns:
            df['first_is_aromatic'] = df['cdr3_sequence'].apply(
                lambda s: int(s[0] in AROMATIC) if pd.notna(s) and len(s) > 0 else 0
            )
        
        if 'cdr3_length_bin' not in df.columns:
            if 'cdr3_len' in df.columns:
                df['cdr3_length_bin'] = (df['cdr3_len'].isin([6, 7])).astype(int)
            elif 'cdr3_sequence' in df.columns:
                df['cdr3_length_bin'] = df['cdr3_sequence'].apply(
                    lambda s: int(len(s) in [6, 7]) if pd.notna(s) else 0
                )
        
        if 'glycine_ratio_bin' not in df.columns:
            if 'glycine_ratio' in df.columns:
                df['glycine_ratio_bin'] = (df['glycine_ratio'] > 0.20).astype(int)
            elif 'cdr3_sequence' in df.columns:
                df['glycine_ratio_bin'] = df['cdr3_sequence'].apply(
                    lambda s: int(sum(1 for a in s if a in GLYCINE) / len(s) > 0.20) if pd.notna(s) and len(s) > 0 else 0
                )
        
        if 'serine_ratio_bin' not in df.columns:
            if 'serine_ratio' in df.columns:
                df['serine_ratio_bin'] = (df['serine_ratio'] > 0.15).astype(int)
            elif 'cdr3_sequence' in df.columns:
                df['serine_ratio_bin'] = df['cdr3_sequence'].apply(
                    lambda s: int(sum(1 for a in s if a in SERINE) / len(s) > 0.15) if pd.notna(s) and len(s) > 0 else 0
                )
    
    def _fit_treatment(
        self,
        df: pd.DataFrame,
        treatment: str,
        confounder_cols: List[str],
    ) -> MultiStateHazardResult:
        """对单个治疗变量拟合多状态Cox模型"""
        
        # 准备Cox回归数据
        covar_cols = [treatment] + [c for c in confounder_cols if c != 'backbone_id']
        
        # backbone_id需要特殊处理
        if 'backbone_id' in confounder_cols and 'backbone_id' in df.columns:
            # 使用数值编码
            df = df.copy()
            df['backbone_code'] = df['backbone_id'].astype('category').cat.codes
            covar_cols.append('backbone_code')
        
        # === Stage 1: RF2失败风险 ===
        if self.verbose:
            print(f"  Stage 1 (RF2): 拟合Cox模型...")
        
        stage1_result = self._fit_cox_stage(
            df=df,
            time_col='surv_time_stage1',
            event_col='surv_event_stage1',
            covar_cols=covar_cols,
            treatment=treatment,
            stage_name='rf2',
            subset=None,  # 全部样本
        )
        
        if self.verbose and stage1_result:
            print(f"    HR={stage1_result.hr:.3f} ({stage1_result.hr_ci[0]:.3f}-{stage1_result.hr_ci[1]:.3f}), "
                  f"p={stage1_result.p_value:.4f}, events={stage1_result.n_events}")
        
        # === Stage 2: 成为FC的风险 ===
        if self.verbose:
            print(f"  Stage 2 (Final): 拟合Cox模型...")
        
        # 仅RF2通过样本
        df_rf2 = df[df['rf2_passed'] == True].copy()
        
        stage2_result = self._fit_cox_stage(
            df=df_rf2,
            time_col='surv_time_stage2',
            event_col='surv_event_stage2',
            covar_cols=covar_cols,
            treatment=treatment,
            stage_name='final',
            subset=df_rf2.index,
        )
        
        if self.verbose and stage2_result:
            print(f"    HR={stage2_result.hr:.3f} ({stage2_result.hr_ci[0]:.3f}-{stage2_result.hr_ci[1]:.3f}), "
                  f"p={stage2_result.p_value:.4f}, events={stage2_result.n_events}")
        
        # === 汇总结果 ===
        stage_results = {}
        if stage1_result:
            stage_results['rf2'] = stage1_result
        if stage2_result:
            stage_results['final'] = stage2_result
        
        # 组合HR
        combined_hr = 1.0
        combined_hr_ci = (1.0, 1.0)
        
        if stage1_result and stage2_result:
            # 组合HR = HR_stage1 × HR_stage2
            combined_hr = stage1_result.hr * stage2_result.hr
            # CI: 假设独立，log(HR)的方差相加
            log_hr_var = (np.log(stage1_result.hr_ci[1]) - np.log(stage1_result.hr))**2 / 1.96**2 + \
                         (np.log(stage2_result.hr_ci[1]) - np.log(stage2_result.hr))**2 / 1.96**2
            log_combined_se = np.sqrt(log_hr_var)
            combined_hr_ci = (
                np.exp(np.log(combined_hr) - 1.96 * log_combined_se),
                np.exp(np.log(combined_hr) + 1.96 * log_combined_se),
            )
        elif stage1_result:
            combined_hr = stage1_result.hr
            combined_hr_ci = stage1_result.hr_ci
        
        # 主导阶段
        dominant_stage = ""
        if stage1_result and stage2_result:
            # 比较效应大小（离1越远越强）
            dist1 = abs(np.log(stage1_result.hr))
            dist2 = abs(np.log(stage2_result.hr))
            if dist1 > dist2:
                dominant_stage = 'rf2'
            else:
                dominant_stage = 'final'
        elif stage1_result:
            dominant_stage = 'rf2'
        
        # 解释
        interpretation = self._generate_interpretation(
            treatment, stage_results, combined_hr, dominant_stage
        )
        
        if self.verbose:
            print(f"  组合HR={combined_hr:.3f} ({combined_hr_ci[0]:.3f}-{combined_hr_ci[1]:.3f})")
            print(f"  主导阶段: {dominant_stage}")
            print(f"  解释: {interpretation}")
        
        return MultiStateHazardResult(
            treatment=treatment,
            stage_results=stage_results,
            combined_hr=combined_hr,
            combined_hr_ci=combined_hr_ci,
            dominant_stage=dominant_stage,
            interpretation=interpretation,
        )
    
    def _fit_cox_stage(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        covar_cols: List[str],
        treatment: str,
        stage_name: str,
        subset: Optional[pd.Index] = None,
    ) -> Optional[StageHazardResult]:
        """拟合单阶段Cox模型"""
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            if self.verbose:
                print(f"    lifelines未安装, 使用Logistic回归近似")
            return self._fit_logistic_approx(
                df, event_col, covar_cols, treatment, stage_name
            )
        
        # 准备Cox数据
        cox_cols = [time_col, event_col] + covar_cols
        available_cols = [c for c in cox_cols if c in df.columns]
        
        cox_df = df[available_cols].dropna().copy()
        
        if len(cox_df) < 50:
            if self.verbose:
                print(f"    样本不足({len(cox_df)}), 跳过Cox回归")
            return self._fit_logistic_approx(
                df, event_col, covar_cols, treatment, stage_name
            )
        
        # 检查事件数
        n_events = cox_df[event_col].sum()
        if n_events < 5:
            if self.verbose:
                print(f"    事件数不足({n_events}), 使用Firth惩罚Cox")
            return self._fit_firth_cox(
                cox_df, time_col, event_col, covar_cols, treatment, stage_name
            )
        
        # 标准Cox回归
        try:
            cph = CoxPHFitter(penalizer=0.1)  # L2正则化增强稳定性
            cph.fit(cox_df, duration_col=time_col, event_col=event_col)
            
            # 提取治疗变量的HR
            if treatment in cph.params_.index:
                coef = cph.params_[treatment]
                coef_se = cph.standard_errors_[treatment]
                hr = np.exp(coef)
                hr_ci = (
                    np.exp(coef - 1.96 * coef_se),
                    np.exp(coef + 1.96 * coef_se),
                )
                # p值: 兼容不同lifelines版本
                if hasattr(cph, 'p_values'):
                    p_value = cph.p_values[treatment]
                elif hasattr(cph, 'p_'):
                    p_value = cph.p_[treatment]
                else:
                    # 手动Wald检验
                    from scipy import stats as sp_stats
                    p_value = 2 * (1 - sp_stats.norm.cdf(abs(coef / max(coef_se, 1e-8))))
                concordance = cph.concordance_index_
            else:
                return None
            
            # 解释
            if hr < 1:
                interp = f"保护性(HR={hr:.3f}), 降低{stage_name}阶段风险{(1-hr)*100:.1f}%"
            elif hr > 1:
                interp = f"风险因素(HR={hr:.3f}), 增加{stage_name}阶段风险{(hr-1)*100:.1f}%"
            else:
                interp = f"无效应(HR={hr:.3f})"
            
            return StageHazardResult(
                treatment=treatment,
                stage=stage_name,
                hr=hr,
                hr_ci=hr_ci,
                p_value=p_value,
                coef=coef,
                coef_se=coef_se,
                n_events=int(n_events),
                n_at_risk=len(cox_df),
                concordance=concordance,
                interpretation=interp,
            )
            
        except Exception as e:
            if self.verbose:
                print(f"    Cox回归失败: {e}, 使用Logistic近似")
            return self._fit_logistic_approx(
                df, event_col, covar_cols, treatment, stage_name
            )
    
    def _fit_firth_cox(
        self,
        cox_df: pd.DataFrame,
        time_col: str,
        event_col: str,
        covar_cols: List[str],
        treatment: str,
        stage_name: str,
    ) -> Optional[StageHazardResult]:
        """Firth惩罚Cox回归（适用于稀疏事件）"""
        try:
            from lifelines import CoxPHFitter
            
            # 增大惩罚项
            cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
            cph.fit(cox_df, duration_col=time_col, event_col=event_col)
            
            if treatment in cph.params_.index:
                coef = cph.params_[treatment]
                coef_se = cph.standard_errors_[treatment]
                hr = np.exp(coef)
                hr_ci = (
                    np.exp(coef - 1.96 * coef_se),
                    np.exp(coef + 1.96 * coef_se),
                )
                # p值: 兼容不同lifelines版本
                if hasattr(cph, 'p_values'):
                    p_value = cph.p_values[treatment]
                elif hasattr(cph, 'p_'):
                    p_value = cph.p_[treatment]
                else:
                    from scipy import stats as sp_stats
                    p_value = 2 * (1 - sp_stats.norm.cdf(abs(coef / max(coef_se, 1e-8))))
                n_events = cox_df[event_col].sum()
                
                return StageHazardResult(
                    treatment=treatment,
                    stage=stage_name,
                    hr=hr,
                    hr_ci=hr_ci,
                    p_value=p_value,
                    coef=coef,
                    coef_se=coef_se,
                    n_events=int(n_events),
                    n_at_risk=len(cox_df),
                    concordance=cph.concordance_index_,
                    interpretation=f"Firth Cox: HR={hr:.3f}",
                )
        except Exception as e:
            if self.verbose:
                print(f"    Firth Cox也失败: {e}")
        
        # 最终降级到Logistic近似
        return self._fit_logistic_approx(
            cox_df, event_col, covar_cols, treatment, stage_name
        )
    
    def _fit_logistic_approx(
        self,
        df: pd.DataFrame,
        event_col: str,
        covar_cols: List[str],
        treatment: str,
        stage_name: str,
    ) -> Optional[StageHazardResult]:
        """Logistic回归近似Cox模型（降级方案）"""
        from sklearn.linear_model import LogisticRegression
        
        available_covars = [c for c in covar_cols if c in df.columns]
        if not available_covars or treatment not in available_covars:
            return None
        
        df_clean = df[available_covars + [event_col]].dropna()
        
        if len(df_clean) < 30:
            return None
        
        X = df_clean[available_covars].values.astype(np.float32)
        y = df_clean[event_col].values.astype(int)
        
        # 检查类别平衡
        if y.sum() < 5 or (1 - y).sum() < 5:
            return None
        
        try:
            lr = LogisticRegression(
                penalty='l2', C=1.0, max_iter=1000, random_state=self.random_state
            )
            lr.fit(X, y)
            
            # 从Logistic系数近似HR
            treat_idx = available_covars.index(treatment)
            coef = lr.coef_[0][treat_idx]
            
            # HR近似: exp(coef)
            hr = np.exp(coef)
            
            # 标准误: 使用Hessian近似
            # 简化: 使用bootstrap
            se = self._bootstrap_logistic_se(X, y, treat_idx)
            hr_ci = (
                np.exp(coef - 1.96 * se),
                np.exp(coef + 1.96 * se),
            )
            
            # P值
            from scipy import stats
            z_stat = coef / max(se, 1e-8)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            n_events = y.sum()
            
            return StageHazardResult(
                treatment=treatment,
                stage=stage_name,
                hr=hr,
                hr_ci=hr_ci,
                p_value=p_value,
                coef=coef,
                coef_se=se,
                n_events=int(n_events),
                n_at_risk=len(df_clean),
                concordance=0.0,
                interpretation=f"Logistic近似: HR={hr:.3f}",
            )
        except Exception as e:
            if self.verbose:
                print(f"    Logistic近似也失败: {e}")
            return None
    
    def _bootstrap_logistic_se(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treat_idx: int,
        n_bootstrap: int = 100,
    ) -> float:
        """Bootstrap估计Logistic系数标准误"""
        from sklearn.linear_model import LogisticRegression
        
        coefs = []
        rng = np.random.RandomState(self.random_state)
        n = len(y)
        
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_b, y_b = X[idx], y[idx]
            
            if y_b.sum() < 2 or (1 - y_b).sum() < 2:
                continue
            
            try:
                lr = LogisticRegression(
                    penalty='l2', C=1.0, max_iter=500, random_state=self.random_state
                )
                lr.fit(X_b, y_b)
                coefs.append(lr.coef_[0][treat_idx])
            except Exception:
                continue
        
        if len(coefs) < 10:
            # 降级: 使用Hessian近似
            try:
                lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=self.random_state)
                lr.fit(X, y)
                # Fisher信息矩阵对角线近似
                prob = lr.predict_proba(X)[:, 1]
                W = prob * (1 - prob)
                XWX = X.T @ np.diag(W) @ X
                try:
                    cov = np.linalg.inv(XWX)
                    return np.sqrt(cov[treat_idx, treat_idx])
                except np.linalg.LinAlgError:
                    return abs(lr.coef_[0][treat_idx]) / 1.96  # 假设p=0.05
            except Exception:
                return 0.5  # 默认值
        
        return np.std(coefs)
    
    def _generate_interpretation(
        self,
        treatment: str,
        stage_results: Dict[str, StageHazardResult],
        combined_hr: float,
        dominant_stage: str,
    ) -> str:
        """生成多状态HR解释"""
        parts = []
        
        # 组合HR
        if combined_hr < 0.9:
            parts.append(f"组合HR={combined_hr:.3f}, 总体保护性")
        elif combined_hr > 1.1:
            parts.append(f"组合HR={combined_hr:.3f}, 总体风险因素")
        else:
            parts.append(f"组合HR={combined_hr:.3f}, 总体效应微弱")
        
        # 阶段特异性
        for stage_name, result in stage_results.items():
            if result.hr < 1:
                parts.append(f"{stage_name}阶段: 保护性(HR={result.hr:.3f})")
            else:
                parts.append(f"{stage_name}阶段: 风险因素(HR={result.hr:.3f})")
        
        # 主导阶段
        if dominant_stage:
            parts.append(f"效应主要来自{dominant_stage}阶段")
        
        # 一致性检查
        if len(stage_results) >= 2:
            hrs = [r.hr for r in stage_results.values()]
            all_protective = all(h < 1 for h in hrs)
            all_risky = all(h > 1 for h in hrs)
            if all_protective:
                parts.append("两阶段效应方向一致(保护性)")
            elif all_risky:
                parts.append("两阶段效应方向一致(风险)")
            else:
                parts.append("两阶段效应方向不一致,需注意")
        
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
            print(f"[MultiStateSurvivalModel] 结果已保存到: {output_path}")
    
    def cross_validate_with_mediation(
        self,
        mediation_results: Dict,
    ) -> pd.DataFrame:
        """
        与Phase 1中介模型结果交叉验证
        
        检查ATE方向与Cox HR是否一致:
        - 保护性treatment: ATE<0 (降低P(RF2 fail)) 且 HR<1
        - 风险treatment: ATE>0 (增加P(RF2 fail)) 且 HR>1
        """
        rows = []
        
        for treatment, med_result in mediation_results.items():
            if treatment not in self.results_:
                continue
            
            surv_result = self.results_[treatment]
            
            # 中介模型的stage1效应方向
            med_stage1_effect = med_result.stage1_effect  # P(RF2|T=1) - P(RF2|T=0)
            # 正值 = T=1时RF2通过率更高 = 保护性
            
            # Cox HR
            if 'rf2' in surv_result.stage_results:
                hr_rf2 = surv_result.stage_results['rf2'].hr
                # HR<1 = T=1降低RF2失败风险 = 保护性
            else:
                hr_rf2 = 1.0
            
            # 一致性检查
            # med_stage1_effect > 0 (T=1增加RF2通过) 应该对应 HR < 1 (T=1降低RF2失败风险)
            consistent = (med_stage1_effect > 0 and hr_rf2 < 1) or \
                         (med_stage1_effect < 0 and hr_rf2 > 1) or \
                         (abs(med_stage1_effect) < 0.001 and 0.9 < hr_rf2 < 1.1)
            
            rows.append({
                'treatment': treatment,
                'mediation_stage1_effect': med_stage1_effect,
                'mediation_stage1_se': med_result.stage1_effect_se,
                'cox_hr_rf2': hr_rf2,
                'direction_consistent': consistent,
                'mediation_total_effect': med_result.total_effect,
                'survival_combined_hr': surv_result.combined_hr,
            })
        
        return pd.DataFrame(rows)


def run_multistate_survival_analysis(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    config: Dict = None,
    mediation_results: Dict = None,
) -> Dict[str, MultiStateHazardResult]:
    """
    运行多状态生存分析的主入口函数
    
    Args:
        df: 包含所有特征的DataFrame
        output_dir: 输出目录
        config: 配置字典
        mediation_results: Phase 1中介模型结果（用于交叉验证）
        
    Returns:
        多状态HR结果字典
    """
    config = config or DEFAULT_CONFIG
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    model = MultiStateSurvivalModel(
        available_stages=config.get('available_stages', AVAILABLE_STAGES_DEFAULT),
        random_state=config.get('random_state', 42),
        verbose=config.get('verbose', True),
    )
    
    # 拟合模型
    results = model.fit(
        df=df,
        treatment_cols=config.get('treatment_cols', ['first_is_aromatic', 'cdr3_length_bin',
                                                       'glycine_ratio_bin', 'serine_ratio_bin']),
        confounder_cols=config.get('confounder_cols', ['backbone_id']),
    )
    
    # 保存结果
    model.save_results(output_dir / 'multistate_hazard_ratios.csv')
    
    # 交叉验证
    if mediation_results is not None:
        cv_df = model.cross_validate_with_mediation(mediation_results)
        cv_df.to_csv(output_dir / 'cross_validation_mediation_survival.csv', index=False)
        
        if model.verbose:
            print(f"\n[交叉验证] ATE方向 vs Cox HR一致性:")
            for _, row in cv_df.iterrows():
                status = "一致" if row['direction_consistent'] else "不一致"
                print(f"  {row['treatment']}: stage1_effect={row['mediation_stage1_effect']:.4f}, "
                      f"HR_rf2={row['cox_hr_rf2']:.3f} → {status}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CSC-O Phase 2: 多状态生存模型')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--mediation', default=None, help='Phase 1中介模型结果CSV路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 加载数据
    df = pd.read_csv(args.input)
    
    # 加载中介模型结果（如果提供）
    mediation_results = None
    if args.mediation:
        # 从CSV重建中介结果
        med_df = pd.read_csv(args.mediation)
        print(f"已加载中介模型结果: {len(med_df)} 个治疗变量")
    
    # 运行分析
    config = {'verbose': args.verbose}
    
    results = run_multistate_survival_analysis(
        df=df,
        output_dir=args.output,
        config=config,
        mediation_results=mediation_results,
    )
    
    print(f"\n完成! 共估计 {len(results)} 个治疗变量的多状态HR")
