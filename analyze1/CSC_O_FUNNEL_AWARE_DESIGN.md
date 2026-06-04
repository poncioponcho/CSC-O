# CSC-O Funnel-Aware Counterfactual Optimization 设计文档

> 版本: v3.0 | 状态: 设计阶段 | 日期: 2026-06-04

## 1. 问题陈述

### 1.1 核心矛盾

当前CSC-O管线(v2.3)优化目标为 **RF2通过率(11.7%)**，但真正目标是 **final_candidate率(0.61%)**。所有因果效应估计和序列生成都指向错误的终点。

### 1.2 数据限制

| 漏斗阶段 | 样本数 | 占比 | 状态 |
|---------|--------|------|------|
| rf2_failed | 9,337 | 88.4% | RF2筛选未通过 |
| rf2_passed | 1,170 | 11.1% | **通过RF2但未被AF3分析(右删失)** |
| final_candidate | 65 | 0.61% | 走完全流程 |

**关键发现**: 不存在"通过RF2但在AF3失败"的中间数据，无法直接估计AF3/Schrödinger阶段特异性转移概率。

### 1.3 已知Bug

v2.3中 `treatment=rf2_passed` (outcome作为treatment)违反因果可操纵性原则，需要重构为proper multi-stage mediation。

---

## 2. 设计决策

### 2.1 数据限制处理

**选择**: 两阶段近似 + `available_stages`参数

```python
# 代码中预留4阶段枚举
ALL_STAGES = ['rf2', 'af3', 'schrodinger', 'final']

# 当前激活阶段（配置控制）
AVAILABLE_STAGES = ['rf2', 'final']  # 后续新数据后改为ALL_STAGES
```

**理由**: 
- 1,170条RF2通过序列在AF3阶段右删失，无法估计中间转移概率
- 两阶段模型: P(final) = P(RF2 pass) × P(final|RF2 pass)
- 代码架构预留完整4阶段，后续只需改配置

### 2.2 稀疏性处理

**选择**: 分解建模 + `method`参数预留扩展

```python
def estimate_final_effect(T, X, method='decomposition'):
    """
    method参数:
    - 'decomposition': P(RF2) × P(final|RF2) 两阶段分解 (当前默认)
    - 'transfer': 迁移学习，用RF2样本预训练+FC样本微调 (FC≥100时启用)
    - 'direct': 直接CausalForestDML (FC≥200时启用)
    """
    if method == 'decomposition':
        return two_stage_decomposition(T, X)
    elif method == 'transfer':
        return transfer_learning_model(T, X)
    elif method == 'direct':
        return direct_causal_forest(T, X)
```

**理由**:
- 65条FC正样本对直接效应估计极度欠功率
- 分解建模: 第一阶段用10,572样本估计P(RF2)，第二阶段用1,235样本+65正例估计P(final|RF2)
- 两阶段分别有足够统计功效

### 2.3 实现顺序

**选择**: 分批实现

| 批次 | Phase | 内容 | 交付物 |
|------|-------|------|--------|
| 第一批 | Phase 1+2 | 因果模型层 | csco_multistage_causal.py, csco_multistate_survival.py |
| 第二批 | Phase 3+4 | 应用层 | csco_funnel_counterfactual.py, csco_funnel_generator.py |
| 第三批 | Phase 5 | 验证框架 | csco_validation_framework.py |

**理由**:
- 第一批打牢因果基础，用交叉验证检查ATE方向
- 第二批直接可用，与v2.3并行跑A/B
- 第三批闭环基础设施

---

## 3. Phase 1: 多阶段中介模型

### 3.1 因果结构

```
Treatment T (可操纵):
├── first_residue_type (aromatic/polar/acidic/basic)
├── cdr3_length (5/6/7)
├── glycine_ratio_bin (>20% vs ≤20%)
└── serine_ratio_bin (>15% vs ≤15%)

Confounders W:
├── ESM-2 fusion embeddings (t12+t30, 1120-dim)
└── backbone_id (one-hot)

Mediator M:
└── rf2_passed (bool)

Outcome Y:
└── final_candidate (bool)

因果路径:
T → M → Y (间接效应)
T → Y (直接效应，绕过RF2)
```

### 3.2 估计方法

**分解建模**:

```
Total Effect on P(final) = TE
  = P(final|T=1) - P(final|T=0)
  = P(RF2|T=1)×P(final|RF2,T=1) - P(RF2|T=0)×P(final|RF2,T=0)

Direct Effect = DE
  = P(final|RF2=1, T=1) - P(final|RF2=1, T=0)  # 控制RF2通过

Indirect Effect = IE
  = TE - DE
```

**实现**:
1. Stage 1: 用全部10,572样本估计 `P(RF2 pass | T, W)` → CausalForestDML
2. Stage 2: 用1,235条RF2通过样本估计 `P(final | T, W, RF2=1)` → LogisticRegression + IPW
3. 合并: `P(final) = P(RF2) × P(final|RF2)`

### 3.3 输出

```python
{
    'treatment': 'first_is_aromatic',
    'total_effect': 0.023,      # P(final)提升2.3pp
    'total_effect_se': 0.008,
    'direct_effect': 0.005,     # 绕过RF2的直接效应
    'direct_effect_se': 0.003,
    'indirect_effect': 0.018,   # 通过RF2的间接效应
    'indirect_effect_se': 0.007,
    'n_stage1': 10572,
    'n_stage2': 1235,
    'n_final': 65
}
```

---

## 4. Phase 2: 多状态生存模型

### 4.1 两阶段近似

```
状态转移图:
  0 (all, n=10572)
    │
    ├─→ 1 (rf2_failed, n=9337)  [事件1: RF2失败]
    │
    └─→ 2 (rf2_passed, n=1235)
          │
          ├─→ 3 (final_candidate, n=65)  [事件2: 成为FC]
          │
          └─→ 4 (censored, n=1170)  [右删失: 未被AF3分析]
```

### 4.2 竞争风险Cox模型

```python
from lifelines import CoxPHFitter

# 阶段1: RF2失败风险
# 风险集: 全部10,572样本
# 事件: rf2_passed=False (n=9,337)
# 删失: rf2_passed=True (n=1,235)
hr_stage1 = fit_cox_stage1(df, treatment, confounders)

# 阶段2: 成为FC的风险
# 风险集: 1,235条RF2通过样本
# 事件: final_candidate=True (n=65)
# 删失: final_candidate=False (n=1,170)
hr_stage2 = fit_cox_stage2(df_rf2_passed, treatment, confounders)
```

### 4.3 输出

```python
{
    'treatment': 'first_is_aromatic',
    'hr_stage1': 0.65,      # RF2失败风险降低35%
    'hr_stage1_ci': (0.52, 0.81),
    'hr_stage2': 0.78,      # 成为FC风险降低22% (在RF2通过者中)
    'hr_stage2_ci': (0.45, 1.35),  # 宽CI因样本少
    'combined_hr': 0.51,    # 组合HR = hr_stage1 × hr_stage2
    'interpretation': '保护性效应主要来自RF2阶段'
}
```

---

## 5. Phase 3: 漏斗感知反事实导航

### 5.1 FunnelAwareCounterfactual类

```python
class FunnelAwareCounterfactual:
    """
    基于P(final)而非P(RF2)的反事实编辑建议
    """
    def __init__(self, causal_model, survival_model, available_stages=['rf2', 'final']):
        self.causal_model = causal_model
        self.survival_model = survival_model
        self.available_stages = available_stages
    
    def suggest_mutation(self, sequence: str, top_k: int = 5) -> List[MutationSuggestion]:
        """
        返回使P(final)最大化的突变建议
        
        输出格式:
        {
            'mutation': 'G→F@pos0',
            'delta_p_rf2': +0.15,
            'delta_p_final_given_rf2': +0.03,
            'delta_p_final': +0.18,  # 总效应
            'confidence': 'high',
            'evidence': 'first_is_aromatic ATE=-6.54, HR_stage1=0.65'
        }
        """
        pass
```

### 5.2 处理65样本限制

- **分解建模**: 不直接估计P(final)，而是估计P(RF2)和P(final|RF2)的乘积
- **重要性加权**: 对65条FC样本赋予更高权重
- **不确定性量化**: 输出置信区间，标注"低置信度"建议

---

## 6. Phase 4: 漏斗感知生成器

### 6.1 升级现有生成器

```python
class FunnelAwareGenerator:
    """
    多目标生成: maximize P(final), not just P(RF2)
    """
    def __init__(self, causal_model, survival_model, strategy_config):
        self.causal_model = causal_model
        self.survival_model = survival_model
        self.strategy_config = strategy_config
    
    def generate(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        生成流程:
        1. Monte Carlo采样候选序列
        2. 用因果模型预测P(RF2)和P(final|RF2)
        3. 计算P(final) = P(RF2) × P(final|RF2)
        4. 按P(final)排序，输出top candidates
        5. 多样性过滤 (edit distance > 3)
        """
        pass
```

### 6.2 阶段特异性约束

从HR_stage1和HR_stage2提取约束:
- `first_is_aromatic`: HR_stage1=0.65 (强保护) → 硬约束
- `glycine_ratio>0.20`: HR_stage1=4.35 (强风险) → 硬约束
- `serine_ratio>0.15`: HR_stage1=1.77 (中等风险) → 软约束

---

## 7. Phase 5: 验证框架

### 7.1 输出格式

```csv
sequence,predicted_p_final,predicted_p_rf2,predicted_p_final_given_rf2,recommended_mutation,ci,diversity_score
ACDYFGH,0.025,0.18,0.14,G→F@pos0,"(0.015,0.035)",4.2
...
```

### 7.2 反馈循环

```python
class ValidationFramework:
    def ingest_experimental_results(self, results_df: pd.DataFrame):
        """
        摄入实验结果，Bayesian更新预测模型
        """
        pass
    
    def ab_test_support(self, control_sequences, treatment_sequences):
        """
        A/B测试支持: 分层随机化 + 功效分析
        """
        pass
```

---

## 8. 成功指标

| 指标 | 目标 | 验证方法 |
|------|------|----------|
| 直接/间接效应 | ≥3个treatment有显著直接效应 | SE < 0.5×效应值 |
| 阶段特异性HR | Gate-specific HR with CI | Cox回归输出 |
| 生成序列质量 | predicted final rate > 20% | 模型预测 |
| 多样性 | diversity score > 4.0 | Shannon熵 |
| 预测性能 | CV-AUC > 0.80 | 5-fold交叉验证 |

---

## 9. 文件结构

```
analyze1/
├── csco_multistage_causal.py      # Phase 1: 中介模型
├── csco_multistate_survival.py    # Phase 2: 多状态生存
├── csco_funnel_counterfactual.py  # Phase 3: 反事实导航
├── csco_funnel_generator.py       # Phase 4: 漏斗感知生成器
├── csco_validation_framework.py   # Phase 5: 验证框架
├── csco_config.py                 # 配置中心(已有，需更新)
└── csco_pipeline.py               # 主管线(已有，需集成)
```

---

## 10. 实现计划

### 第一批: 因果模型层 (Phase 1+2)

**交付物**:
1. `csco_multistage_causal.py` — 中介模型
2. `csco_multistate_survival.py` — 两阶段Cox模型

**验证**:
- 交叉验证检查ATE方向与Cox HR一致性
- 例: `first_is_aromatic` 应为保护性(ATE<0, HR<1)

**预计工作量**: 核心实现 + 单元测试 + 集成测试

### 第二批: 应用层 (Phase 3+4)

**交付物**:
1. `csco_funnel_counterfactual.py` — 反事实导航
2. `csco_funnel_generator.py` — 升级生成器
3. 新版 `design_strategy_v3.0.json`

**验证**:
- 生成1000条序列，检查P(final)分布
- 与v2.3并行跑A/B对比

### 第三批: 验证框架 (Phase 5)

**交付物**:
1. `csco_validation_framework.py` — 反馈循环
2. A/B测试基础设施

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 65样本稀疏性 | 直接效应估计不稳定 | 分解建模 + 置信区间标注 |
| 右删失数据 | 无法估计中间阶段HR | 两阶段近似 + 预留扩展接口 |
| ATE方向错误 | 整个漏斗感知概念失效 | 交叉验证 + Cox HR交叉检查 |
| 与v2.3冲突 | 破坏现有功能 | 独立模块 + 并行运行 |

---

## 12. 待用户确认

- [ ] Phase 1中介模型的治疗变量选择是否完整?
- [ ] 分解建模的两阶段估计是否接受?
- [ ] 第一批交付后是否需要完整测试报告?
