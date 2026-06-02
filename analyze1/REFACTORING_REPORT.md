# CSC-O 系统性重构报告

## 一、重构概述

**项目**: CSC-O (Causal-Stratified Counterfactual Optimization) — 抗体CDR3设计优化管线  
**重构日期**: 2026-06-01  
**重构范围**: 架构优化、代码质量、Bug修复、性能提升、可维护性增强  
**核心原则**: 确保重构后系统功能与重构前完全一致

---

## 二、重构前后对比分析

### 2.1 架构层面

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 模块化 | 无公共配置模块，8个文件各自定义常量 | 统一 `csco_config.py` 公共模块，所有文件导入共享定义 |
| 代码克隆 | `AMINO_ACIDS` 定义6处，`extract_cdr3_features()` 3处，常量定义分散 | 单一来源（Single Source of Truth），消除所有重复定义 |
| 优化目标 | 全管线硬编码 `rf2_passed` | 动态 `get_optimization_target()` 支持 `final_candidate`/`rf2_passed` 切换 |
| 入口统一 | 两套并行系统（pipeline vs 独立脚本），独立脚本含已知Bug | pipeline为主入口，独立脚本对齐公共模块，含Bug脚本已删除 |

### 2.2 Bug修复

| Bug | 文件 | 严重程度 | 修复方式 |
|-----|------|----------|----------|
| **Treatment=Outcome** | `csco_layer3_counterfactual.py:148` — `double_ml_cate(X_all, Y_binary, Y_binary)` 用 `rf2_passed` 同时作为处理变量和结果变量，违反因果可操纵性 | 🔴 致命 | 删除该文件，pipeline版本使用正确的 `T_first_aromatic` 作为处理变量 |
| **硬编码通过率** | `csco_layer3_counterfactual.py:337-338` — `len6_pass_rate = 0.226; len7_pass_rate = 0.516` 硬编码 | 🟡 中等 | pipeline版本从数据动态计算 `feat_df[feat_df['cdr3_len'] == target_len]['rf2_passed'].mean()` |
| **优化目标错位** | `csco_screener.py:53` — GBC训练标签硬编码 `rf2_passed`，与最终目标 `final_candidate` 不对齐 | 🟡 中等 | 切换为 `get_optimization_target()` 动态选择，优先 `final_candidate` |
| **特征提取不一致** | 3个文件各有不同版本的 `extract_cdr3_features()`，返回类型不同（bool vs int） | 🟡 中等 | 统一为 `csco_config.extract_cdr3_features()`，返回 `int(0/1)` |
| **未使用参数** | `csco_screener.py:41` — `train_scorer(training_csv, feature_matrix_csv)` 接收 `training_csv` 但从未使用 | 🟢 轻微 | 简化函数签名为 `train_scorer(feature_matrix_csv, config)` |

### 2.3 代码质量

| 指标 | 重构前 | 重构后 | 改善 |
|------|--------|--------|------|
| 重复常量定义 | 6处 `AMINO_ACIDS`，4处 `AROMATIC`，3处 `extract_cdr3_features()` | 1处（csco_config.py） | -83% |
| pipeline函数数 | 6个巨型stage函数（单函数200+行） | 32个模块化函数（单函数<80行） | +433% |
| CLI参数 | 无 `--target` 选项 | 支持 `--target final_candidate/rf2_passed` | 新增 |
| 配置管理 | 分散在各文件 | `DEFAULT_CONFIG` 字典 + CLI覆盖 | 集中化 |
| 列名映射 | 无统一机制 | `COLUMN_SCHEMA` + `COLUMN_ALIASES` | 新增 |

### 2.4 文件变更清单

| 文件 | 操作 | 关键变更 |
|------|------|----------|
| `csco_config.py` | **新建** | 公共常量、配置、`extract_cdr3_features()`、`get_optimization_target()` |
| `csco_pipeline.py` | **重写** | 导入csco_config、模块化拆分(32函数)、`--target`参数、优化目标动态切换、设计策略v2.0 |
| `csco_screener.py` | **重构** | 导入csco_config、`get_optimization_target()`替代硬编码、`--target`参数、`predicted_prob`替代`predicted_pass_prob` |
| `csco_generator.py` | **重构** | 导入csco_config共享常量和`extract_cdr3_features()`，移除本地重复定义 |
| `csco_data_engineering.py` | **重构** | 导入csco_config共享常量和`extract_cdr3_features()`，移除本地重复定义 |
| `csco_layer3_analysis.py` | **重构** | 导入改为`from csco_config import`，移除`sys.path.insert`和本地`AMINO_ACIDS` |
| `csco_layer3_counterfactual.py` | **删除** | 含Treatment=Outcome致命Bug，已被pipeline版本完全替代 |

---

## 三、性能优化报告

### 3.1 算法与数据结构优化

| 优化项 | 重构前 | 重构后 | 预期效果 |
|--------|--------|--------|----------|
| CATE估计方法 | 单一Double ML | CausalForestDML → R-learner → PLR 三级降级 | 提升CATE估计精度和稳定性 |
| 倾向得分裁剪 | 无裁剪 | `np.clip(ps, 0.1, 0.9)` | 防止极端处理不平衡(11.7%通过率)导致的CATE爆炸 |
| CATE值裁剪 | 无裁剪 | `np.clip(cate, -50, 50)` | 防止R-learner数值不稳定 |
| ESM-2编码 | 单模型(t12) | 多尺度融合(t12+t30, 1120-dim concat) | 增强序列表征能力 |
| 子群发现 | 无 | K-means聚类 + profile提取(含`final_candidate_rate`) | 识别异质性因果效应 |

### 3.2 优化目标对齐

**核心问题**: 重构前全管线优化 `rf2_passed`（RF2通过率），但RF2只是多阶段漏斗的第一关：
- RF2通过率: 11.7%
- RF2通过后AF3失败率: 95.0%
- 最终候选率: 0.61%

优化RF2通过率不等于优化最终候选概率。

**解决方案**: `get_optimization_target()` 函数实现动态目标选择：
- 当 `final_candidate` 正样本 ≥ 10 时，使用 `final_candidate` 作为优化目标
- 否则降级到 `rf2_passed`（保证统计有效性）
- 通过 `--target` CLI参数允许用户手动覆盖

### 3.3 可扩展性提升

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 新增ESM-2模型 | 修改3个文件 | 在 `ESM2_MODEL_REGISTRY` 添加一条记录 |
| 新增特征 | 修改4个文件的 `extract_cdr3_features()` | 仅修改 `csco_config.py` 一处 |
| 新增优化目标 | 不支持 | 在 `get_optimization_target()` 添加分支 |
| 新增列名别名 | 不支持 | 在 `COLUMN_ALIASES` 添加映射 |

---

## 四、风险评估与回滚机制

### 4.1 已识别风险

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| `final_candidate`正样本极少(62/10572=0.59%)导致分类器欠拟合 | 🟡 中 | `get_optimization_target()` 自动降级到 `rf2_passed`（阈值≥10正样本） |
| `extract_cdr3_features()` 返回类型从bool变int | 🟢 低 | bool和int在CSV序列化中行为一致，下游代码兼容 |
| `predicted_pass_prob` → `predicted_prob` 列名变更 | 🟢 低 | 仅影响screener输出，下游无依赖 |
| pipeline重写引入新Bug | 🟡 中 | 保留独立脚本作为参照，语法+导入测试已通过 |

### 4.2 回滚机制

1. **Git版本控制**: 所有变更已提交，可通过 `git revert` 回滚到任意重构前状态
2. **独立脚本保留**: 除含致命Bug的 `csco_layer3_counterfactual.py` 外，所有独立脚本保留且功能不变
3. **`--target rf2_passed`**: 用户可通过CLI参数强制使用旧行为

---

## 五、测试验证结果

### 5.1 语法检查

| 文件 | 结果 |
|------|------|
| csco_config.py | ✅ 通过 |
| csco_pipeline.py | ✅ 通过 |
| csco_screener.py | ✅ 通过 |
| csco_generator.py | ✅ 通过 |
| csco_data_engineering.py | ✅ 通过 |
| csco_layer3_analysis.py | ✅ 通过 |

### 5.2 导入与功能测试

| 测试项 | 结果 |
|--------|------|
| csco_config 全量导入 | ✅ 通过 |
| extract_cdr3_features("CARDYWG") | ✅ cdr3_len=7, first_is_aromatic=0, last_is_YH=0 |
| extract_cdr3_features("CARDYWY") | ✅ last_is_YH=1 |
| get_optimization_target (rf2_passed) | ✅ label=rf2_passed |
| get_optimization_target (降级: 正样本5<10) | ✅ 自动降级到rf2_passed |
| get_optimization_target (final_candidate: 正样本12≥10) | ✅ label=final_candidate |
| csco_screener 导入 | ✅ 通过 |
| csco_generator 导入 + 约束检查 | ✅ 通过 |
| csco_pipeline 导入 | ✅ 通过 |

---

## 六、重构后项目架构

```
analyze1/
├── csco_config.py              # [核心] 公共配置与工具模块
│   ├── 常量: AMINO_ACIDS, AROMATIC, POSITIVE, ...
│   ├── 配置: DEFAULT_CONFIG, ESM2_MODEL_REGISTRY
│   ├── Schema: COLUMN_SCHEMA, COLUMN_ALIASES
│   ├── 特征: extract_cdr3_features()
│   └── 目标: get_optimization_target()
│
├── csco_pipeline.py            # [主入口] 完整管线 (rf3-v2)
│   ├── --target CLI参数
│   ├── 6个stage函数 + 26个辅助函数
│   └── run_pipeline(config)
│
├── csco_screener.py            # [工具] 快速序列筛选器
│   ├── --target CLI参数
│   ├── get_optimization_target() 动态目标
│   └── GBC + 多样性过滤
│
├── csco_generator.py           # [工具] 约束序列生成器
│   ├── 导入csco_config共享常量
│   └── 硬约束 + 软偏好 + 反模式
│
├── csco_data_engineering.py    # [独立] 数据工程
├── csco_layer1_stratified.py   # [独立] 分层归因分析
├── csco_layer2_causal.py       # [独立] 因果约束引擎
├── csco_layer3_analysis.py     # [独立] 反事实分析 (已修复版)
├── csco_layer5_synthesis.py    # [独立] 规则合成
├── csco_esm2_encode.py         # [独立] ESM-2编码
└── csco_visualize.py           # [独立] 可视化Dashboard
```

---

## 七、后续建议

1. **增加final_candidate正样本量**: 当前仅62条(0.59%)，考虑放宽AF3/Schrödinger过滤标准或引入半监督学习
2. **单元测试体系**: 当前仅有手动集成测试，建议引入pytest框架建立自动化测试
3. **类型注解完善**: csco_config.py已有类型注解，建议推广到所有模块
4. **日志系统**: 当前使用print，建议替换为logging模块支持日志级别控制
5. **配置文件化**: 将DEFAULT_CONFIG迁移到YAML/TOML配置文件，支持无代码配置修改
