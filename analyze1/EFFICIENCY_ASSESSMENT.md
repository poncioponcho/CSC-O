# CSC-O 项目效率评估与优化方案

## 一、当前效率基线

### 1.1 运行时间基线
基于 10572 条序列的实际运行数据：

| 阶段 | 耗时 | 占比 | 瓶颈 |
|------|------|------|------|
| data_engineering | ~30s | <1% | iterrows遍历 |
| layer1_stratified | ~60s | <1% | Cox PH拟合 |
| layer2_causal | ~120s | 1% | PC算法组合爆炸 |
| layer3_esm2_encode | ~1800s (30min) | 32% | GPU推理 |
| layer3_counterfactual | ~3180s (53min) | 57% | R-learner ×8 treatment |
| layer5_synthesis | ~0.1s | <1% | - |
| **总计** | **~5190s (87min)** | 100% | |

核心瓶颈：layer3_counterfactual (57%) + layer3_esm2_encode (32%) = 89%

### 1.2 资源利用率基线
- CPU: 单核为主（inner_n_jobs=1），8核利用率<15%
- GPU: ESM-2编码时使用，其余时间闲置
- 内存: 峰值~2GB（ESM-2模型+嵌入矩阵）
- 磁盘: ~100MB输出文件

### 1.3 代码质量基线
- 总代码行数: ~1400行（csco_pipeline.py）
- 函数数: 25+
- 重复代码: _double_ml_cate 中3种方法有大量重复
- 测试覆盖: 0%（无单元测试）
- 类型标注: 无
- 文档字符串: 部分

## 二、优化方案一：算法并行化与异步流水线

### 2.1 优化目标
- 总运行时间从 87min → 25-35min（2.5-3.5×加速）
- CPU利用率从 <15% → 60-80%

### 2.2 技术实现路径

#### 2.2.1 ESM-2与因果推断流水线化
当前：ESM-2编码 → 等待完成 → 因果推断（串行）
优化：ESM-2编码的同时，用阶段2的ATE结果预计算不需要嵌入的部分

#### 2.2.2 Multi-treatment CATE 真正并行化
当前：8个treatment串行（因之前joblib死锁问题回退）
优化：使用 multiprocessing.Process 替代 joblib.Parallel（避免fork死锁），每个子进程独立运行，inner_n_jobs=1

```python
from multiprocessing import Process, Queue

def _worker(t_col, feat_df, X_all, Y_pae, cate_method, result_queue):
    # 独立进程，无GIL问题
    cate_t, tstat_t, se_t = _double_ml_cate(X_conf, T_binary, Y_pae, method=cate_method, inner_n_jobs=1)
    result_queue.put({...})

processes = []
result_queue = Queue()
for t_col in treatment_cols:
    p = Process(target=_worker, args=(t_col, ...))
    p.start(); processes.append(p)
for p in processes: p.join()
```

预期：8个treatment并行 → 从53min → ~7min（8核时）

#### 2.2.3 位置特异性CATE并行化
当前：13个位置 × 20个氨基酸 = ~260次Ridge回归串行
优化：按位置分组并行

### 2.3 所需资源
- 8核CPU服务器
- 16GB内存（每个子进程~2GB）
- 开发工时：3-5天

### 2.4 潜在风险与规避
| 风险 | 概率 | 影响 | 规避措施 |
|------|------|------|---------|
| 多进程内存溢出 | 中 | 高 | 限制并发数=min(n_treatment, cpu_count-1) |
| 子进程异常无响应 | 低 | 高 | 添加timeout参数，超时自动终止 |
| 结果顺序不一致 | 低 | 低 | 使用OrderedDict或按treatment名排序 |
| 共享内存竞争 | 低 | 中 | 使用Queue通信，避免共享状态 |

### 2.5 实施优先级
**P0 - 高** | 预期工时：3-5天 | 预期提升：2.5-3.5×

## 三、优化方案二：算法效率优化与近似计算

### 3.1 优化目标
- layer3_counterfactual 从 53min → 10-15min（3-5×加速）
- 保持CATE估计精度损失<5%

### 3.2 技术实现路径

#### 3.2.1 增量式CATE计算
当前：每个treatment独立完整计算
优化：共享第一阶段的Y_residual和T_residual计算

```python
# 所有treatment共享的Y模型只需训练一次
m_y_shared = LGBMRegressor().fit(X, Y)
Y_resid_shared = Y - m_y_shared.predict(X)

# 每个treatment只需训练T模型
for t_col in treatment_cols:
    m_t = LGBMClassifier().fit(X, T)
    T_resid = T - m_t.predict_proba(X)[:, 1]
    # 共享Y_resid，只重新计算T_resid
```

预期：减少5/8的Y模型训练时间

#### 3.2.2 子采样+外推
当前：10572条全量数据训练R-learner
优化：用5000条子样本训练，全量数据推断CATE

```python
sample_idx = np.random.choice(len(X), 5000, replace=False)
m_y.fit(X[sample_idx], Y[sample_idx])  # 训练用子样本
cate_all = hetero_model.predict(X)       # 推断用全量
```

预期：训练时间减少~50%，精度损失<3%

#### 3.2.3 LightGBM早停
当前：固定n_estimators=100
优化：使用early_stopping_rounds=20

```python
m_y = LGBMRegressor(n_estimators=500, early_stopping_rounds=20)
m_y.fit(X_tr, Y_tr, eval_set=[(X_te, Y_te)])
```

预期：实际使用树数量通常30-60棵，加速2-3×

#### 3.2.4 反事实建议生成优化
当前：2000条序列 × 260种突变 = 520000次查表
优化：向量化查表，避免逐条循环

### 3.3 所需资源
- 无额外硬件需求
- 开发工时：2-3天

### 3.4 潜在风险与规避
| 风险 | 概率 | 影响 | 规避措施 |
|------|------|------|---------|
| 子采样导致CATE偏差 | 中 | 中 | 使用分层抽样保持treatment分布 |
| 早停导致欠拟合 | 低 | 中 | 设置n_estimators上限=500 |
| 共享Y模型假设不成立 | 低 | 低 | 对比独立训练与共享训练的CATE差异 |

### 3.5 实施优先级
**P1 - 中高** | 预期工时：2-3天 | 预期提升：3-5×

## 四、优化方案三：代码架构重构与工程化

### 4.1 优化目标
- 代码可维护性提升（模块化、类型化、可测试）
- 新用户上手时间从 2小时 → 30分钟
- Bug修复时间从 数小时 → 数分钟

### 4.2 技术实现路径

#### 4.2.1 模块化拆分
当前：csco_pipeline.py 1400行单文件
目标：
```
csco/
├── __init__.py
├── config.py          # 配置和常量
├── data_adapter.py    # 数据加载和适配
├── stages/
│   ├── data_engineering.py
│   ├── stratified.py
│   ├── causal.py
│   ├── esm2_encode.py
│   ├── counterfactual.py
│   └── synthesis.py
├── methods/
│   ├── cox.py
│   ├── pc_algorithm.py
│   ├── ate.py
│   ├── double_ml.py
│   └── position_cate.py
├── utils/
│   ├── plotting.py
│   └── io.py
└── cli.py             # 命令行入口
```

#### 4.2.2 类型标注
为所有函数添加类型标注，提升IDE支持和代码可读性：
```python
def _double_ml_cate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    method: str = "causal_forest",
    inner_n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

#### 4.2.3 单元测试
```python
# tests/test_double_ml.py
def test_plr_returns_correct_shapes():
    X, T, Y = make_test_data(n=200)
    cate, tstat, se = _double_ml_cate(X, T, Y, method="plr")
    assert cate.shape == (200,)
    assert tstat.shape == (200,)
    assert se.shape == (200,)

def test_r_learner_cate_bounded():
    cate, _, _ = _double_ml_cate(X, T, Y, method="r_learner")
    assert cate.min() >= -50
    assert cate.max() <= 50
```

#### 4.2.4 配置文件化
将硬编码参数移到 YAML 配置文件：
```yaml
# config/default.yaml
esm2:
  model: esm2_t12_35M_UR50D
  batch_size: 4
  fusion: concat

causal:
  cate_method: r_learner
  n_folds: 5
  inner_n_jobs: 1

counterfactual:
  top_n: 2000
  n_neighbors: 5
```

#### 4.2.5 日志系统
替换 print 为 logging：
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"[{i+1}/{total}] {t_col}: cate_mean={result['cate_mean']:.3f}")
```

### 4.3 所需资源
- 无额外硬件需求
- 开发工时：5-7天

### 4.4 潜在风险与规避
| 风险 | 概率 | 影响 | 规避措施 |
|------|------|------|---------|
| 重构引入新bug | 高 | 高 | 先写测试再重构；逐步迁移 |
| 接口兼容性破坏 | 中 | 中 | 保留旧接口作为wrapper |
| 过度工程化 | 中 | 低 | 遵循YAGNI原则，只拆分必要模块 |

### 4.5 实施优先级
**P2 - 中** | 预期工时：5-7天 | 预期提升：可维护性3-5×，上手效率4×

## 五、优化方案对比总结

| 维度 | 方案一：并行化 | 方案二：算法优化 | 方案三：架构重构 |
|------|--------------|----------------|----------------|
| 核心目标 | 运行速度 | 算法效率 | 工程质量 |
| 预期加速 | 2.5-3.5× | 3-5× (单阶段) | 间接（开发效率） |
| 硬件需求 | 8核CPU | 无 | 无 |
| 开发工时 | 3-5天 | 2-3天 | 5-7天 |
| 风险等级 | 中 | 低 | 中 |
| 实施优先级 | P0 | P1 | P2 |
| 依赖关系 | 独立 | 独立 | 建议在方案一二之后 |

## 六、推荐实施路线

### 第一阶段（1周）：方案一 + 方案二
1. 先实施方案二的LightGBM早停和子采样（1天）
2. 再实施方案一的multiprocessing并行化（2天）
3. 集成测试和性能基准（1天）
4. 预期效果：87min → 15-20min

### 第二阶段（1周）：方案三
1. 模块化拆分（2天）
2. 类型标注 + 测试（2天）
3. 配置文件化 + 日志系统（1天）

### 第三阶段（持续）：迭代优化
1. 根据实际运行数据持续调优
2. 添加更多因果推断方法
3. 支持更多蛋白质设计场景
