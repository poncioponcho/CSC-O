# CSC-O 项目审计报告

> **审计角色**：审计分析师 + 文档工程师
> **审计范围**：v2.0 ~ v2.3 全版本数据、代码、报告
> **判定标准**：无原始数据支撑、未通过统计显著性检验、或依赖已废弃假设却被表述为确定性的结论 = "虚构结论"
> **声明**：所有修复方案以diff/伪代码/步骤清单形式输出，需人工审核后方可应用于生产环境

---

# 交付物1：指标提升效果报告

## 1. 数据质量审计表

### 1.1 数据完整性检查

| 检查项 | 状态 | 详情 |
|--------|------|------|
| 原始CSV本地可用性 | ❌ 不可用 | `.gitignore`排除`*.csv`，本地仅存代码与报告，CSV仅在远程服务器 |
| v2.0结果数据 | ❌ 不可用 | `output_server_v2.0/` 已被排除 |
| v2.1结果数据 | ❌ 不可用 | `output_server_v2.1/` 已被排除 |
| v2.2结果数据 | ❌ 不可用 | `output_server_v2.2/` 已被排除 |
| v2.3结果数据 | ⚠️ 部分可用 | `output_server_v2.3/design_strategy.json` 存在，但CSV结果不在本地 |
| 生成器测试输出 | ✅ 可用 | `output_v2.3_test/generated_sequences.csv` (791条) 已在上一轮验证中生成 |

### 1.2 基于代码审计的数据质量推断

| 数据集 | 样本量 | final_candidate正样本 | 正样本率 | 缺失值风险 | 分布漂移风险 |
|--------|--------|----------------------|---------|-----------|-------------|
| Q02223原始 | 10,572 | 65 | 0.61% | 低（DataAdapter有_fill_missing_columns） | 基线 |
| v2.0 RF2通过 | 1,235 | 65 | 5.26% | 中（rf2_passed_filter被覆盖重算） | 低 |
| v2.1 RF2通过 | 2,908 | 65 | 2.24% | **高**（新增1673条边缘通过，FC=0） | **高** |
| v2.2 RF2通过 | 2,512 | 65 | 2.59% | 中（lddt=0.86收紧） | 中 |
| v2.3 RF2通过 | ~2,300 | 65 | ~2.83% | 中（白名单[F,W,Y]） | 中 |

### 1.3 关键数据质量问题

**Q1: rf2_passed_filter 无条件覆盖**

```python
# csco_pipeline.py L197-198
# 始终基于原始指标重新计算rf2_passed_filter（使用可配置阈值）
self.df['rf2_passed_filter'] = lddt_ok & pae_ok & rmsd_ok
```

- **影响**：原始数据中可能由下游信息（AF3/Schrodinger反馈）修正的rf2_passed_filter标注会被丢弃
- **风险等级**：中 — 若原始标注是纯RF2指标则冗余；若包含下游信息则数据丢失

**Q2: CDR3序列未去重**

- **影响**：同一CDR3在不同backbone/run中出现时被当作独立样本
- **受影响环节**：Cox回归（权重放大）、ATE估计（SE偏小）、success_templates（高频序列主导）

**Q3: final_candidate正样本极度稀疏**

- 10,572条中仅65条final_candidate=True (0.61%)
- 任何基于此标签的统计推断功效极低
- 白名单/阈值搜索在小子集上容易过拟合

---

## 2. 漏洞排查清单

### V1: 二元Treatment ATE无统计检验

| 字段 | 内容 |
|------|------|
| **问题描述** | 二元treatment（如first_is_aromatic, last_is_YH）的ATE估计返回SE=0, t_stat=0, p_value=0 |
| **代码位置** | `csco_pipeline.py` L605-606 |
| **影响指标** | ATE估计不可检验，无法判断效应是否显著异于零；design_strategy.json中的ate_estimates字段含不可靠数据 |
| **修复diff** | ```python\n# 修复前:\nate_results.append({'treatment': tv, 'outcome': outcome_binary,\n    'ATE': ate, 'SE': 0, 't_stat': 0, 'p_value': 0,\n    'CI_lower': 0, 'CI_upper': 0})\n\n# 修复后: 使用bootstrap估计SE\nn_boot = 1000\nboot_ates = []\nfor _ in range(n_boot):\n    idx = rng.choice(len(Y), len(Y), replace=True)\n    boot_ates.append(_compute_ipw_ate(Y[idx], T[idx], ipw[idx]))\nse = np.std(boot_ates)\nt_stat = ate / se if se > 0 else 0\np_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))\nci_lower = ate - 1.96 * se\nci_upper = ate + 1.96 * se\n``` |
| **验证用例** | 对first_is_aromatic做bootstrap ATE，检查SE>0且p_value在合理范围；与连续treatment的OLS ATE交叉验证 |

### V2: 全量数据拟合策略参数（Look-ahead Bias）

| 字段 | 内容 |
|------|------|
| **问题描述** | 白名单、软阈值、反模式检测、salvage_edits全部在全量数据上拟合，无train/test分割 |
| **代码位置** | `csco_pipeline.py` L1134-1317 (stage_layer5_synthesis) |
| **影响指标** | design_strategy.json中的规则在训练数据上有效，在新数据上可能大幅失效；FC率提升预期不可信 |
| **修复diff** | ```python\n# 修复方案: 80/20分割 + 交叉验证\nfrom sklearn.model_selection import StratifiedShuffleSplit\nsss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,\n    random_state=config.get('seed', 42))\ntrain_idx, val_idx = next(sss.split(feat_df, feat_df[target_col]))\ntrain_df = feat_df.iloc[train_idx]\nval_df = feat_df.iloc[val_idx]\n\n# 在train_df上推导策略参数\nstrategy = _derive_strategy(train_df, config)\n\n# 在val_df上验证\nval_pass = _evaluate_strategy(val_df, strategy)\nprint(f\"验证集FC率: {val_pass['fc_rate']:.2%}\")\n``` |
| **验证用例** | 5折交叉验证，每折在训练集推导策略、在验证集计算FC率；若验证集FC率与训练集差异>50%则存在过拟合 |

### V3: 白名单阈值硬编码且不一致

| 字段 | 内容 |
|------|------|
| **问题描述** | first_whitelist用RF2通过率>0.70，last_whitelist用>0.08；first有FC=0排除，last无；阈值无统计依据 |
| **代码位置** | `csco_pipeline.py` L1155 (first: 0.70), L1171 (last: 0.08) |
| **影响指标** | last_whitelist形同虚设（0.08阈值几乎所有残基都入选）；首尾白名单逻辑不对称 |
| **修复diff** | ```python\n# 统一白名单逻辑：基于Fisher精确检验\nfrom scipy.stats import fisher_exact\nfor aa in all_residues:\n    contingency = pd.crosstab(\n        feat_df['first_residue'] == aa,\n        feat_df[target_col]\n    )\n    odds_ratio, p_value = fisher_exact(contingency)\n    if p_value < 0.05 and odds_ratio > 1.0:\n        whitelist.append(aa)\n``` |
| **验证用例** | 对v2.3数据应用Fisher检验，验证[F,W,Y]的p值<0.05且OR>1；验证被排除残基的p值>0.05或OR<1 |

### V4: 混杂变量不足导致遗漏变量偏误

| 字段 | 内容 |
|------|------|
| **问题描述** | ATE估计仅控制backbone_id，但cdr3_len/aromatic_ratio/hydrophobic_ratio等强混杂未控制 |
| **代码位置** | `csco_pipeline.py` L600-601 |
| **影响指标** | ATE估计可能有偏；glycine_ratio ATE=+5.41可能部分由遗漏的hydrophobic_ratio混杂解释 |
| **修复diff** | ```python\n# 修复前:\nconfounders = ['backbone_id']\n\n# 修复后: 使用csco_config.py中已定义的CONFOUNDER_COLS\nfrom csco_config import CONFOUNDER_COLS\nconfounders = [c for c in CONFOUNDER_COLS if c != treatment_var and c in feat_df.columns]\n# CONFOUNDER_COLS = ['backbone_id', 'aromatic_ratio',\n#                    'hydrophobic_ratio', 'positive_ratio']\n``` |
| **验证用例** | 对glycine_ratio分别用仅backbone_id和完整CONFOUNDER_COLS做ATE估计，比较差异；若差异>20%则确认遗漏偏误 |

### V5: 分层ATE无多重比较校正

| 字段 | 内容 |
|------|------|
| **问题描述** | 对6个长度×8个treatment=48个检验，用p<0.05筛选但未做Bonferroni/FDR校正 |
| **代码位置** | `csco_pipeline.py` L617-666 |
| **影响指标** | 假阳性率可能高达1-(1-0.05)^48≈91%；length_specific_preferences可能包含虚假规则 |
| **修复diff** | ```python\n# 修复: 添加BH-FDR校正\nfrom statsmodels.stats.multitest import multipletests\nall_pvalues = [r['p_value'] for r in stratified_results]\n_, q_values, _, _ = multipletests(all_pvalues, method='fdr_bh')\nfor i, r in enumerate(stratified_results):\n    r['q_value'] = q_values[i]\n    r['significant_fdr'] = q_values[i] < 0.05\n``` |
| **验证用例** | 比较p<0.05与FDR q<0.05的显著结果数量；若FDR后显著数大幅减少，确认假阳性问题 |

### V6: 白名单/阈值搜索无样本量下限

| 字段 | 内容 |
|------|------|
| **问题描述** | 某首残基可能仅3条数据，通过率100%就入选白名单，统计上极不可靠 |
| **代码位置** | `csco_pipeline.py` L1153-1167 |
| **影响指标** | 白名单可能包含小样本偶然通过的残基 |
| **修复diff** | ```python\n# 修复: 添加最小样本量要求\nMIN_SAMPLE_FOR_WHITELIST = 30  # 至少30条数据\nfirst_res = first_res[first_res['total'] >= MIN_SAMPLE_FOR_WHITELIST]\n``` |
| **验证用例** | 检查v2.3数据中F/W/Y各自的样本量是否>=30；检查被排除残基中是否有小样本高通过率的 |

---

## 3. 改进方案

### 3.1 数据层面

| 改进项 | 具体步骤 | 预期效果 |
|--------|---------|---------|
| **D1: CDR3去重** | 在DataAdapter加载后按cdr3_sequence去重，保留final_candidate=True的优先 | 消除重复采样导致的SE偏小和权重放大 |
| **D2: 训练/验证分割** | 80/20分层分割，策略参数仅在训练集推导 | 消除look-ahead bias，提供可信的泛化性能估计 |
| **D3: 缺失值系统性审计** | 对rf2_pred_lddt/rf2_interaction_pae等关键列统计缺失率，按backbone_id分组 | 识别数据缺失模式，避免NaN填充引入偏误 |
| **D4: rf2_passed_filter保留原始** | 新增列rf2_passed_filter_v2而非覆盖原始列 | 保留原始标注信息，便于版本对比 |

### 3.2 蛋白质序列层面

| 改进项 | 具体步骤 | 预期效果 |
|--------|---------|---------|
| **P1: 统一白名单方法论** | 用Fisher精确检验替代硬编码阈值，首尾残基使用同一逻辑 | 白名单选择有统计依据，消除0.70/0.08不一致 |
| **P2: 位置特异性编码替代one-hot** | 用ESM-2残基级嵌入替代first_residue/last_residue的离散编码 | 捕捉残基间相似性（F≈W≈Y），提升CATE估计精度 |
| **P3: 生物约束集成** | 在generator中添加CDR3空间约束（如Ramachandran偏好、二面角可行性） | 减少结构不可行的序列，提升RF2通过率 |
| **P4: 多样性-质量Pareto优化** | 在screener中同时优化soft_score和序列多样性，而非先质量后去重 | 避免高质量但低多样性的序列集 |

---

## 4. 效果对比

### 4.1 各版本RF2通过与FC率对比

| 版本 | RF2阈值 | 首残基白名单 | RF2通过数 | FC数 | FC率 | vs基线 |
|------|---------|------------|----------|------|------|--------|
| v2.0 | lddt≥0.85, PAE≤12 | 无限制 | 1,235 | 65 | 0.61% | 基线 |
| v2.1 | lddt≥0.85, PAE≤12 | 无限制 | 2,908 | 65 | 0.61% | +0% (虚假通过) |
| v2.2 | lddt≥0.86, PAE≤10 | 6种 | 2,512 | 65 | 0.61% | +0% (过滤改善但FC未增) |
| v2.3 | lddt≥0.86, PAE≤10 | [F,W,Y] | ~2,300 | 65 | ~2.83% | +3.6× (仅RF2层) |

### 4.2 关键发现

```
final_candidate率变化 (全量数据视角)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v2.0  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.61%  (65/10572)
v2.1  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.61%  (65/10572) ← 虚假通过未增加FC
v2.2  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.61%  (65/10572) ← 阈值收紧未增加FC
v2.3  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~2.8%  (RF2层)   ← 白名单提升RF2层通过率

注: final_candidate=65条在所有版本中均未变化，
    因为FC是外部标注的金标准，管线从未修改它。
    v2.3的"提升"仅体现在RF2筛选层的通过率，
    尚未有新序列走完AF3→Schrodinger→Desmond全流程。
```

### 4.3 核心结论

**final_candidate=65条在v2.0~v2.3所有版本中均未变化**。所有版本的"优化"仅影响RF2筛选层的通过率和序列质量分布，尚未有新生成序列走完AF3→Schrodinger→Desmond全流程并产生新的final_candidate。因此，**任何声称"FC率从0.61%提升至X%"的结论，若X≠0.61%，均为虚构结论**。

---

# 交付物2：历史问题总结报告

## 1. 虚构结论清单

| 编号 | 虚构结论 | 出处 | 表述方式 | 实际数据支撑 |
|------|---------|------|---------|-------------|
| FC-1 | "FC率从0.61%提升至3.5-6.0%" | VERSION_RELEASE.md | 确定性目标声明 | ❌ 无任何新FC序列产生 |
| FC-2 | "保守估计FC率≈1.6%" | csco_optimization_report.md §10.2 | 等号呈现估计值 | ❌ 基于未验证的线性叠加假设 |
| FC-3 | "乐观估计FC率≈3.1%" | csco_optimization_report.md §10.2 | 等号呈现估计值 | ❌ 同上 |
| FC-4 | "FC率→3.5-6.0%" | csco_optimization_report.md §10.2 | 箭头暗示确定性路径 | ❌ 无独立验证集验证 |
| FC-5 | "RF2通过率预计提升30-50%" | csco_optimization_report.md §9建议1 | "预计"但以精确范围呈现 | ❌ 无定量模型推导 |
| FC-6 | "FC率预计提升0.2-0.4个百分点" | csco_optimization_report.md §9建议1 | "预计" | ❌ 无推导依据 |
| FC-7 | "截短后RF2通过率从<1%提升至10-20%" | csco_optimization_report.md §3.3,§9 | "预计" | ⚠️ 仅基于ATE差值粗估，未验证 |
| FC-8 | "整体RF2通过率预计从11.7%提升至15-20%" | csco_optimization_report.md §9建议2 | "预计" | ❌ 无推导模型 |
| FC-9 | "AF3通过率预计从5.0%提升至8-12%" | csco_optimization_report.md §9建议6 | "预计" | ❌ 无推导依据 |
| FC-10 | "总运行时间从87min→25-35min(2.5-3.5×加速)" | EFFICIENCY_ASSESSMENT.md §2.1 | 精确范围呈现 | ❌ 理论预期，无基准测试 |
| FC-11 | "精度损失<5%"/"精度损失<3%" | EFFICIENCY_ASSESSMENT.md §3.1,§3.2.2 | 性能保证声明 | ❌ 无实验支撑 |
| FC-12 | "预期里程碑：FC率从0.61%提升至1.5-2.5%" | csco_optimization_report.md §11 | 里程碑形式 | ❌ 无验证数据 |
| FC-13 | "预期里程碑：FC率从1.5-2.5%提升至3.0-5.0%" | csco_optimization_report.md §11 | 里程碑形式 | ❌ 无验证数据 |
| FC-14 | "可维护性3-5×，上手效率4×" | EFFICIENCY_ASSESSMENT.md §4.5 | 精确倍数 | ❌ 指标无量化定义 |

## 2. 根因分析

### FC-1/FC-2/FC-3/FC-4: FC率提升预期

| 维度 | 分析 |
|------|------|
| **产生路径** | 管线分析→识别保护因子→设计策略→**假设策略在新序列上同样有效**→线性叠加效应→得出FC率预期 |
| **数据错误** | 无。基础数据（Cox HR、ATE）本身有统计检验 |
| **统计方法误用** | ① 效应线性叠加假设未验证（SUTVA违反）；② 全量数据拟合无独立验证；③ 正样本仅65条，任何子组分析统计功效不足 |
| **确认偏误** | 从"RF2通过率提升"直接跳到"FC率提升"，忽略了RF2→AF3→Schrodinger→Desmond的漏斗衰减 |
| **影响范围** | VERSION_RELEASE.md（核心版本声明）、csco_optimization_report.md §9-11（建议与路线图） |

### FC-5~FC-9: RF2/AF3通过率提升预期

| 维度 | 分析 |
|------|------|
| **产生路径** | ATE估计→"glycine_ratio ATE=+5.41"→假设降低glycine可降低PAE 5.41单位→推算RF2通过率提升 |
| **数据错误** | ATE是条件平均处理效应，不是PAE的绝对变化量；ATE=+5.41意味着glycine_ratio每增加1单位PAE增加5.41，但glycine_ratio从0.2降到0.15仅变化0.05，实际PAE变化≈0.27而非5.41 |
| **统计方法误用** | 将ATE系数直接当作绝对效应量，忽略了treatment的实际变化范围 |
| **确认偏误** | 选择性关注大ATE值，忽略小效应或负效应 |
| **影响范围** | csco_optimization_report.md §3.3, §4.2, §9 |

### FC-10~FC-14: 效率与工程指标预期

| 维度 | 分析 |
|------|------|
| **产生路径** | 理论分析→假设完美线性加速→得出加速比→以精确数值呈现 |
| **数据错误** | 无基准测试数据 |
| **统计方法误用** | 不适用（非统计问题） |
| **确认偏误** | 忽略了并行开销、I/O瓶颈、内存限制等现实约束 |
| **影响范围** | EFFICIENCY_ASSESSMENT.md全文 |

## 3. 信任修复机制：三重验证门

### 数据门（原始数据可溯源）

```
┌─────────────────────────────────────────────┐
│ 数据门检查清单                                │
├─────────────────────────────────────────────┤
│ □ 数据文件路径是否明确标注？                    │
│ □ 数据版本（v2.0/v2.1/v2.2/v2.3）是否标注？    │
│ □ 样本量是否报告？                             │
│ □ 缺失值比例是否报告？                         │
│ □ 数据预处理步骤是否可复现？                    │
└─────────────────────────────────────────────┘
```

**实施方式**：所有报告中的数值结论必须附带数据溯源标签，格式：`[数据源:文件名:版本:行号]`

### 方法门（统计检验前置）

```
┌─────────────────────────────────────────────┐
│ 方法门检查清单                                │
├─────────────────────────────────────────────┤
│ □ 是否报告了SE/置信区间？                      │
│ □ p值是否做了多重比较校正？                     │
│ □ 样本量是否满足统计功效要求？                   │
│ □ 效应量是否与treatment变化范围匹配？            │
│ □ 是否有独立验证集？                           │
└─────────────────────────────────────────────┘
```

**实施方式**：无统计检验的结论必须标注`[未验证]`标签，不得以确定性语气表述

### 结论门（置信区间强制披露）

```
┌─────────────────────────────────────────────┐
│ 结论门检查清单                                │
├─────────────────────────────────────────────┤
│ □ 预期值是否附带置信区间？                      │
│ □ "预计"/"预期"是否与"已验证"明确区分？          │
│ □ 组合效应是否标注了叠加假设？                   │
│ □ 是否有回滚预案？                             │
└─────────────────────────────────────────────┘
```

**实施方式**：所有预期值格式改为`预期值 (95%CI: [下限, 上限], 依据: 方法/数据源)`

## 4. 典型案例

### 案例1: "FC率从0.61%提升至3.5-6.0%"

| 环节 | 内容 |
|------|------|
| **原始陈述** | "优化目标: final_candidate率从0.61%提升至3.5-6.0%" (VERSION_RELEASE.md) |
| **数据反证** | ① v2.0~v2.3所有版本中final_candidate=65条从未变化；② 3.5-6.0%意味着370-634条FC，需要新生成序列走完RF2→AF3→Schrodinger→Desmond全流程；③ 截至v2.3，无任何新生成序列完成全流程验证 |
| **根因** | 将"策略设计目标"表述为"优化效果"；将RF2层通过率提升外推至FC率提升，忽略漏斗衰减 |
| **纠正后结论** | "优化目标: 通过白名单[F,W,Y]和阈值收紧(lddt≥0.86,PAE≤10)，将RF2筛选层通过序列的质量提升（平均PAE从6.27降至6.06），但final_candidate率仍为0.61%。FC率提升至3.5-6.0%为未验证的预期目标，需待新生成序列完成全流程验证后方可确认。" |

### 案例2: "glycine_ratio ATE=+5.41 → 降低glycine可大幅改善PAE"

| 环节 | 内容 |
|------|------|
| **原始陈述** | "glycine_ratio ATE=+5.41, p<1e-40" → 推论"降低glycine_ratio可降低PAE约5.41单位" |
| **数据反证** | ① ATE=+5.41意味着glycine_ratio每增加1个单位(0→1)，PAE增加5.41；② 实际glycine_ratio范围约0~0.4，中位数~0.1；③ 将glycine_ratio从0.2降至0.15（变化0.05），预期PAE变化仅0.05×5.41≈0.27，而非5.41；④ 且此ATE仅控制backbone_id，未控制hydrophobic_ratio等混杂 |
| **根因** | 将回归系数（单位变化效应）直接当作绝对效应量，忽略了treatment的实际变化范围 |
| **纠正后结论** | "glycine_ratio对PAE的ATE=+5.41 (p<1e-40, 95%CI未报告, 仅控制backbone_id)。在glycine_ratio典型变化范围0.05内，预期PAE变化约0.27单位。此估计可能因遗漏hydrophobic_ratio等混杂变量而有偏。" |

---

# 交付物3：过程资产与方向漂移记录

> **统一声明**：本交付物记录的问题与final_candidate指标无直接因果关系，但可能影响试验可复现性与未来迭代成本。

| # | 问题描述 | 产生背景 | 类别 | 当前状态 | 与核心试验耦合度 | 后续建议 |
|---|---------|---------|------|---------|---------------|---------|
| PA-1 | PC算法方向推断最后一行`abs(corr)>0`几乎永远True | 因果发现阶段实现不完整，domain_constraints绕过了方向学习 | 技术债 | 已处理(用domain_constraints替代) | 低 | 冻结，当前DAG仅用于可视化 |
| PA-2 | csco_layer5_synthesis.py旧版硬编码预期效果数字 | v1独立脚本，后整合入pipeline但旧文件未删除 | 废弃分支 | 挂起 | 低 | 废弃旧文件，避免混淆 |
| PA-3 | ESM-2编码重复计算同一vh_sequence | 缓存机制基于文件存在性而非序列去重 | 技术债 | 未开始 | 中 | 添加序列级缓存，减少50%+编码时间 |
| PA-4 | _search_soft_threshold贪心搜索依赖候选列表顺序 | 策略参数推导采用简单贪心，未做网格搜索 | 架构试探 | 已处理(可用但次优) | 中 | 改为网格搜索或贝叶斯优化 |
| PA-5 | CausalForestDML嵌套并行死锁 | macOS fork-without-exec + joblib嵌套 | 技术债 | 已处理(inner_n_jobs=1) | 低 | 冻结，服务器端用n_jobs=-1 |
| PA-6 | 反事实建议对重复CDR3重复生成 | 未按cdr3_sequence去重 | 技术债 | 未开始 | 中 | 去重后再生成建议 |
| PA-7 | length_specific_preferences的ATE→规则映射全硬编码 | 策略合成阶段需要将连续ATE映射为离散规则 | 架构试探 | 已处理(可用但脆弱) | 中 | 改为数据驱动的阈值选择 |
| PA-8 | funnel_stage用final_candidate推断存在循环定义 | _infer_funnel_stage的优先级链中FC在最前 | 技术债 | 已处理(仅影响可视化) | 低 | 冻结，funnel_stage不进入模型 |
| PA-9 | VPN连接不稳定导致服务器部署频繁中断 | 远程开发环境依赖VPN | 废弃分支 | 挂起 | 低 | 迁移至稳定网络或容器化部署 |
| PA-10 | 数据脱敏后GitHub推送含残留敏感信息 | 首次脱敏不完整 | 技术债 | 已处理(二次扫描) | 低 | 定期审计git历史 |
| PA-11 | rf2_passed_filter无条件覆盖原始标注 | v2.1为解决阈值不生效问题而强制重算 | 技术债 | 已处理(但丢失原始信息) | 中 | 保留原始列，新增v2列 |
| PA-12 | last_whitelist无FC率过滤，与first_whitelist逻辑不一致 | 白名单推导时首尾使用了不同标准 | 技术债 | 未开始 | 中 | 统一首尾白名单方法论 |
| PA-13 | 生成器template变异可能产生非白名单首残基 | 30%概率从template变异，template含G/D/S等首残基 | 架构试探 | 已处理(check_hard_constraints兜底) | 低 | 过滤template列表，仅保留白名单首残基的template |
| PA-14 | 评估协议缺失：无独立测试集评估策略效果 | 全量数据拟合策略，无hold-out验证 | 待验证假设 | 未开始 | 高 | 必须建立train/val分割评估协议 |

---

# 交付物4：待决策事项清单

| 优先级 | # | 问题描述 | 决策建议 | 理由 | 风险评级 | 预估工时 | 潜在副作用 | 回滚预案 |
|--------|---|---------|---------|------|---------|---------|-----------|---------|
| **1** | PA-14 | 评估协议缺失 | **纳入** | 无独立验证集则所有策略效果声明不可信；FC率提升预期无法验证 | 高 | 4h | 对feat_df的80/20分割可能因FC正样本仅65条导致验证集FC=0 | 降低分割比为90/10或使用5折CV |
| **2** | V2 | 全量数据拟合策略参数 | **纳入** | look-ahead bias是FC率预期不可信的直接原因 | 高 | 6h | 策略参数在验证集上可能表现差，需重新调优 | 恢复全量数据拟合，但标注"未验证" |
| **3** | V3 | 白名单阈值硬编码不一致 | **纳入** | last_whitelist形同虚设(0.08阈值)，首尾逻辑不对称 | 中 | 3h | 收紧last_whitelist可能减少生成序列数量 | 恢复原始阈值 |
| **4** | V5 | 分层ATE无多重比较校正 | **纳入** | 48个检验的假阳性率~91%，length_specific_preferences可能含虚假规则 | 中 | 2h | FDR校正后部分长度特定规则可能不显著 | 保留原始p值筛选，标注"未校正" |
| **5** | PA-11 | rf2_passed_filter覆盖原始 | **纳入** | 丢失原始标注信息，无法回溯对比 | 中 | 1h | 新增列不影响现有逻辑 | 删除新增列即可 |
| **6** | V1 | 二元Treatment ATE无统计检验 | **纳入** | design_strategy.json中ate_estimates含不可靠数据 | 中 | 3h | bootstrap增加计算时间(~5min) | 使用原始SE=0 |
| **7** | V4 | 混杂变量不足 | **纳入** | ATE可能有偏，影响策略参数可信度 | 中 | 4h | 控制更多混杂后ATE可能变小/变号 | 恢复仅backbone_id |
| **8** | PA-12 | last_whitelist无FC率过滤 | **纳入** | 与first_whitelist逻辑不一致 | 中 | 1h | 可能排除部分尾残基 | 恢复无FC过滤 |
| **9** | PA-4 | 贪心搜索次优 | 冻结 | 当前可用，优化收益不确定 | 低 | 8h | - | - |
| **10** | PA-7 | ATE→规则映射硬编码 | 冻结 | 当前可用，重构成本高 | 低 | 12h | - | - |
| **11** | PA-3 | ESM-2重复编码 | 冻结 | 性能问题不影响结果正确性 | 低 | 4h | - | - |
| **12** | PA-2 | 旧版layer5脚本 | **废弃** | 已被pipeline整合，保留易混淆 | 低 | 0.5h | - | - |
| **13** | PA-9 | VPN不稳定 | **废弃** | 基础设施问题，非项目核心 | 低 | - | - | - |
| **14** | V6 | 白名单无样本量下限 | **纳入** | 小样本残基偶然通过导致白名单不可靠 | 中 | 1h | 可能排除样本量<30的残基 | 降低下限至10 |

### 优先级排序说明

排序公式：`优先级 = 风险评级(高=3,中=2,低=1) × 耦合度(高=3,中=2,低=1)`

| 排名 | # | 优先级得分 | 决策 |
|------|---|----------|------|
| 1 | PA-14 | 3×3=9 | 纳入 |
| 2 | V2 | 3×3=9 | 纳入 |
| 3 | V3 | 2×2=4 | 纳入 |
| 4 | V5 | 2×2=4 | 纳入 |
| 5 | PA-11 | 2×2=4 | 纳入 |
| 6 | V1 | 2×2=4 | 纳入 |
| 7 | V4 | 2×2=4 | 纳入 |
| 8 | PA-12 | 2×2=4 | 纳入 |
| 9 | V6 | 2×2=4 | 纳入 |
| 10 | PA-4 | 1×2=2 | 冻结 |
| 11 | PA-7 | 1×2=2 | 冻结 |
| 12 | PA-3 | 1×2=2 | 冻结 |
| 13 | PA-2 | 1×1=1 | 废弃 |
| 14 | PA-9 | 1×1=1 | 废弃 |

### "纳入"项对final_candidate计算链路的潜在副作用

| # | 副作用 | 回滚预案 |
|---|--------|---------|
| PA-14 | 80/20分割后训练集FC正样本可能仅52条，统计功效进一步降低 | 改用5折CV或90/10分割 |
| V2 | 策略参数在验证集上FC率可能远低于训练集，需重新调优 | 恢复全量拟合+标注"未验证" |
| V3 | 收紧last_whitelist后生成序列减少10-20% | 恢复0.08阈值 |
| V5 | FDR校正后length_specific_preferences规则减少，生成多样性可能下降 | 保留原始p<0.05筛选 |
| V1 | bootstrap SE可能使部分ATE不显著，影响ate_estimates字段 | 保留原始SE=0+标注 |
| V4 | 控制更多混杂后ATE变小，策略参数可能需调整 | 恢复仅backbone_id |
| V6 | 样本量<30的残基被排除，白名单可能缩小 | 降低下限至10 |
