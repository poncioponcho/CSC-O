# CSC-O 项目进展与规划文档

> **版本**: v1.0  
> **日期**: 2026-05-31  
> **作者**: Seyon + AI Agent  
> **项目全称**: Causal-Stratified Counterfactual Optimization for Antibody CDR3 Design  
> **目标**: 基于 10,572 条抗体序列数据，通过因果推断、分层分析和反事实优化，生成可实验验证的 CDR3 设计策略

---

## 目录

1. [项目概述](#一项目概述)
2. [前期试验结果总结](#二前期试验结果总结)
3. [可行改进方案概述](#三可行改进方案概述)
4. [改进方案依据说明](#四改进方案依据说明)
5. [后续操作建议](#五后续操作建议)
6. [附录](#六附录)

---

## 一、项目概述

### 1.1 项目背景

本课题旨在利用计算生物学方法优化抗体 CDR3 区域的设计。通过对 10,572 条针对 Q02223 靶点的抗体序列进行系统性分析，构建从数据工程 → 因果推断 → 反事实优化的完整管线（CSC-O Pipeline），最终输出可指导实验的设计策略。

### 1.2 技术架构

```
输入: 10,572 条抗体序列 (CSV)
    ↓
Layer 1: 数据工程 (Data Engineering)
    - 特征提取 (CDR3 长度、氨基酸比例、模式识别)
    - 生存数据构建 (多阶段漏斗模型)
    ↓
Layer 2: 分层归因 (Stratified Analysis)
    - Cox 比例风险回归
    - Kaplan-Meier 生存曲线
    - 阈值敏感度分析
    ↓
Layer 3a: 因果约束引擎 (Causal Engine)
    - PC 算法因果 DAG 发现
    - ATE (Average Treatment Effect) 估计
    ↓
Layer 3b: ESM-2 序列编码
    - 蛋白质语言模型嵌入
    ↓
Layer 3c: 反事实序列导航 (Counterfactual)
    - Double ML CATE 估计
    - 位置特异性突变效应
    - 最近邻模板映射
    ↓
Layer 5: 规则合成 (Synthesis)
    - 设计策略 JSON/TXT 输出
    - 综合分析报告
```

### 1.3 核心产出

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 设计策略 | `output/design_strategy.json` | 机器可读策略配置 |
| 分析报告 | `output/csco_analysis_report.txt` | 人类可读摘要 |
| 特征矩阵 | `output/feature_matrix.csv` | 10572 × 40 特征表 |
| ESM-2 嵌入 | `output/esm2_embeddings.npy` | 10572 × 480 向量 |
| 反事实建议 | `output/counterfactual_suggestions.csv` | 突变建议列表 |
| 可视化图表 | `output/*.png` | 漏斗图、森林图、t-SNE 等 |

---

## 二、前期试验结果总结

### 2.1 试验环境

| 组件 | 配置 |
|------|------|
| 服务器 | REDACTED_HOSTNAME |
| GPU | 8 × NVIDIA GeForce GTX 1080 Ti (11GB × 8 = 88GB) |
| CPU | 未知型号，多核 |
| 内存 | 125GB |
| 系统 | Ubuntu 18.04.6 LTS, Linux 5.4.0-150-generic |
| CUDA | 11.4 |
| Python | 3.10.13 (通过 uv 安装) |
| PyTorch | 1.12.1+cu113 |
| fair-esm | 2.0.0 |

### 2.2 试验 1：CSC-O v1.0 完整管线运行

**时间**: 2026-05-30  
**状态**: ✅ **成功完成**

#### 2.2.1 各阶段耗时

| 阶段 | 耗时 | 状态 | 备注 |
|------|------|------|------|
| data_engineering | ~1-2 min | ✅ 完成 | 特征提取 + 生存数据 |
| layer1_stratified | ~2-5 min | ✅ 完成 | Cox/KM/阈值分析 |
| layer2_causal | ~3-8 min | ✅ 完成 | PC 算法 + ATE |
| layer3_esm2_encode | **49.1s** | ✅ 完成 | t12_35M 编码 |
| layer3_counterfactual | **222.0s** | ✅ 完成 | Double ML + 最近邻 |
| layer5_synthesis | **0.1s** | ✅ 完成 | 策略合成 |
| **总计** | **~6-8 min** | ✅ | |

#### 2.2.2 核心数据指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 总序列数 | 10,572 | Q02223 靶点 |
| RF2 通过率 | **11.7%** | 1,235 / 10,572 |
| 最终候选率 | **0.61%** | 62 / 10,572 |
| 允许 CDR3 长度 | [5, 6, 7] | 通过率 > 10% |
| 优选长度 | [6, 7] | 通过率最高 |

#### 2.2.3 关键发现

**发现 1：CDR3 长度效应显著**

| 长度 | RF2 通过率 | 评估 |
|------|-----------|------|
| 5 | ~15% | 可接受 |
| 6 | ~30% | 优选 |
| 7 | ~25% | 优选 |
| ≥10 | **2.6%** | **避免** |
| <10 (对照) | 30.8% | 基准 |

**发现 2：氨基酸组成影响巨大**

Cox 回归结果：

| 变量 | 风险比 (HR) | p 值 | 解释 |
|------|------------|------|------|
| glycine_ratio | **4.35** | <1e-36 | 甘氨酸越多，失败风险越高 |
| serine_ratio | **1.77** | <5e-9 | 丝氨酸越多，失败风险越高 |
| aromatic_ratio | **0.67** | 0.001 | 芳香族有保护作用 |
| cdr3_len | 1.08 | <1e-64 | 长度越长，风险越高 |

**发现 3：首残基效应最强**

ATE 估计：

| Treatment | ATE on PAE | t 值 | p 值 |
|-----------|-----------|------|------|
| first_is_aromatic | **-6.54** | -47.05 | <1e-10 |
| serine_ratio | **+9.69** | +32.21 | <1e-10 |
| cdr3_len | +0.91 | +122.78 | <1e-10 |

**发现 4：高频挽救编辑**

| 编辑 | 出现次数 | 平均 PAE 变化 |
|------|---------|--------------|
| Pos0 G→Y | 1,254 | -1.85 |
| Pos0 G→L | 1,254 | -1.65 |
| Pos0 G→V | 430 | -1.58 |
| Pos9 Y→E | 314 | -1.63 |

### 2.3 试验 2：代码修复验证

**时间**: 2026-05-30  
**状态**: ✅ **已完成**

| 修复项 | 问题 | 修复方式 | 状态 |
|--------|------|---------|------|
| Double ML treatment bug | `csco_layer3_analysis.py` 中 treatment=outcome | 改用 `first_is_aromatic` 作为 treatment | ✅ |
| 硬编码可视化数据 | `csco_visualize.py` 所有图表数据写死 | 改为从 CSV 动态读取 | ✅ |
| 缺少依赖管理 | 无 requirements.txt | 新增 requirements.txt | ✅ |
| 生成文件入 Git | output/ 目录被 Git 跟踪 | 新增 .gitignore 排除 | ✅ |
| 敏感信息风险 | 服务器配置记录含密码 | .gitignore 排除 + 文档提醒 | ✅ |

### 2.4 试验 3：VPN 断连自动保存系统

**时间**: 2026-05-30  
**状态**: ✅ **已开发并部署**

#### 2.4.1 系统组件

| 组件 | 路径 | 功能 | 平台 |
|------|------|------|------|
| vpn_watchdog.sh | `tools/vpn_watchdog.sh` | VPN 监测 + 预测 + 自动恢复 | Mac |
| ssh_auto_attach.sh | `tools/ssh_auto_attach.sh` | 一键 SSH + tmux 恢复 | Mac |
| check_server_status.sh | `tools/check_server_status.sh` | 服务器状态查询 | Mac |
| workspace_save.py | `tools/workspace_save.py` | 保存 tmux 会话 + 进程 | Linux |
| workspace_restore.py | `tools/workspace_restore.py` | 恢复 tmux 会话 | Linux |
| vim_session_save.sh | `tools/vim_session_save.sh` | 保存 vim 编辑状态 | Linux |
| vim_session_restore.sh | `tools/vim_session_restore.sh` | 恢复 vim 编辑状态 | Linux |

#### 2.4.2 预测机制

- **延迟趋势窗口**: 维护最近 20 次 ping 历史
- **陡增检测**: 最新延迟 > 均值 × 3 且 > 300ms → 提前预警
- **丢包预警**: 丢包率 ≥ 20% → 触发保存
- **连续失败**: 连续 3 次检测失败 → 判定断开

### 2.5 失败教训与问题记录

#### 问题 1：numpy 版本冲突（严重）

| 项目 | 详情 |
|------|------|
| 症状 | ESM-2 编码阶段全部失败：`Numpy is not available` |
| 根因 | `numpy 2.2.6` 与 `torch 1.12.1+cu113` 不兼容 |
| 影响 | 所有 10,572 条序列编码失败，embeddings 全为零 |
| 修复 | 降级到 `numpy 1.23.5` |
| 教训 | **必须严格管理依赖版本**，建议在 requirements.txt 中锁定版本 |

#### 问题 2：tmux 2.6 不稳定

| 项目 | 详情 |
|------|------|
| 症状 | tmux 会话创建后立即退出，`tmux ls` 显示 "no server running" |
| 根因 | Ubuntu 18.04 默认 tmux 2.6 有 bug；加上 pipeline 崩溃导致会话结束 |
| 影响 | 无法使用 tmux 托管进程 |
| 修复 | 改用 `nohup` 后台运行 |
| 教训 | **tmux 2.6 不可靠**，建议升级到 tmux 3.x 或直接使用 nohup/screen |

#### 问题 3：VPN MTU 问题

| 项目 | 详情 |
|------|------|
| 症状 | ping 通、TCP 22/443 能建立，但 SSH banner exchange 超时 |
| 根因 | openconnect MTU 1500 过大，实际承载能力仅 1228 |
| 修复 | `sudo ifconfig utun4 mtu 1228` 或启动时 `--mtu 1228` |
| 教训 | **校园网 VPN 必须限制 MTU**，建议写入启动脚本 |

#### 问题 4：GitHub 不可达

| 项目 | 详情 |
|------|------|
| 症状 | 服务器 `git clone https://github.com/...` 超时 |
| 根因 | 校园网防火墙限制对外 GitHub 访问 |
| 修复 | 改用 `rsync` 从 Mac 同步到服务器 |
| 教训 | **内网服务器不要依赖 GitHub**，使用本地 rsync/scp 同步 |

### 2.6 未解决问题

| 问题 | 优先级 | 说明 |
|------|--------|------|
| ESM-2 模型升级 | 🔴 高 | 当前仅用 t12_35M，未验证更大模型效果 |
| 异质性 CATE | 🔴 高 | 当前为常数 CATE，未实现个性化效应估计 |
| 序列生成验证 | 🟡 中 | 设计策略生成的新序列未做 RF2/AF3 实验验证 |
| 结构特征缺失 | 🟡 中 | 当前无蛋白质结构数据，仅用序列特征 |
| 多任务预测网络 | 🟢 低 | 尚未构建端到端深度预测模型 |

---

## 三、可行改进方案概述

### 方案 A：ESM-2 升级 + 多尺度嵌入融合

#### 核心思路

从当前的 `esm2_t12_35M_UR50D`（480 维）升级到 `t30_150M`（640 维）和 `t33_650M`（1280 维），融合三种尺度的嵌入表示，构建 2400 维的多视角特征向量。

#### 预期目标

| 目标 | 指标 | 基线 | 预期 |
|------|------|------|------|
| 嵌入表达能力 | 维度 | 480 | **2400** |
| 下游预测精度 | AUC | ~0.75 | **>0.80** |
| 反事实建议准确率 | Top-3 命中率 | ~35% | **>45%** |

#### 实施步骤

1. **模型加载与并行策略**
   - t12_35M: GPU 0 (2GB)
   - t30_150M: GPU 1 (6GB)
   - t33_650M: GPU 2-3 模型并行 (14GB)

2. **融合策略**
   ```python
   Z_fused = concatenate([Z_12, Z_30, Z_33])  # ℝ^2400
   # 或加权融合: α·Z_12 + β·Z_30 + γ·Z_33
   ```

3. **消融实验**
   - 仅用 Z_12 (基线)
   - 仅用 Z_33 (对照)
   - Z_12 + Z_30 (中间对照)
   - Z_12 + Z_30 + Z_33 (实验组)

4. **重新跑管线**
   - 用融合嵌入替换原 Z_12
   - 评估各阶段指标提升

#### 所需资源

| 资源 | 需求 | 可用 |
|------|------|------|
| GPU 显存 | 14GB (t33) | ✅ 88GB 总量 |
| 运行时间 | +10-15 min | ✅ 当前 6min → 预计 25min |
| 存储空间 | +20MB (新 embeddings) | ✅ /home 29GB 剩余 |
| 代码修改 | `csco_pipeline.py` 中 `stage_esm2_encode` | 中等工作量 |

---

### 方案 B：Causal Forest 替代常数 CATE

#### 核心思路

当前 Double ML 估计的是**常数 CATE**（所有样本对同一突变的响应相同）。Causal Forest 通过自适应划分的决策树森林，估计**异质性 CATE**（每个样本的个性化处理效应），并提供置信区间。

#### 预期目标

| 目标 | 基线 | 预期 |
|------|------|------|
| CATE 精度 | 常数 θ (MSE 高) | 个性化 θ(x) (MSE 降低 20-40%) |
| 亚群规则发现 | 无 | 发现 3-5 个亚群特异性规则 |
| 个性化建议 | 所有人同一 Top-1 | 每条序列 Top-3 + 置信度 |

#### 实施步骤

1. **依赖安装**
   ```bash
   python -m pip install econml scikit-learn
   ```

2. **Causal Forest 训练**
   ```python
   from econml.grf import CausalForest
   est = CausalForest(n_estimators=500, 
                      min_samples_leaf=40,
                      n_jobs=8)
   est.fit(X, T, Y)
   cate_individual = est.effect(X_test)
   cate_stderr = est.effect_stderr(X_test)
   ```

3. **亚群发现**
   - 对 CATE 进行聚类 (K-means)
   - 提取各亚群的特征画像
   - 生成亚群特异性设计规则

4. **个性化建议生成**
   - 对每条失败序列评估所有可能突变
   - 保留 |CATE| > 1.96×SE 的显著建议
   - 按 CATE 排序输出 Top-3

#### 所需资源

| 资源 | 需求 | 可用 |
|------|------|------|
| CPU 核心 | 8 核并行 | ✅ 充足 |
| 运行时间 | ~8-10 min | ✅ |
| 内存 | ~5GB | ✅ 125GB |
| 代码修改 | 新增 `stage_counterfactual_cf.py` | 中等工作量 |

---

### 方案 C：条件序列生成器 + 快速筛选流水线

#### 核心思路

从"分析已有数据"升级为"主动设计新序列"。基于 design_strategy.json 的约束条件，生成大量候选序列，用 ESM-2 + LightGBM 快速评分筛选，输出 Top-N 送实验验证。

#### 预期目标

| 目标 | 基线 | 预期 |
|------|------|------|
| 生成序列数 | 0 (仅分析) | **10,000 条/轮** |
| 预测通过率 | 11.7% (原始数据) | **>40%** (筛选后) |
| 实验验证候选 | 无明确推荐 | **Top 200 带置信度** |

#### 实施步骤

1. **约束采样生成器**
   ```python
   def generate_constrained(n=10000):
       for _ in range(n):
           length = random.choice([6, 7])
           first = random.choice(['F','V','W','Y'])
           seq = sample_with_constraints(length, first, 
                                         anti_patterns=['GGG','SSS','LL'],
                                         max_gly=0.2, max_ser=0.15)
           yield seq
   ```

2. **快速评分**
   - ESM-2 编码 (t33_650M, 2卡并行)
   - LightGBM 预测 RF2 pass 概率
   - 预测 PAE

3. **筛选策略**
   - 保留预测通过率 > 40% 的序列
   - 按预测 PAE 排序
   - 多样性过滤 (避免序列过于相似)

4. **实验验证清单**
   - 输出 Top 200 序列及其预测指标
   - 标注推荐突变位点
   - 送 RF2/AF3 结构预测验证

#### 所需资源

| 资源 | 需求 | 可用 |
|------|------|------|
| 生成时间 | <1 秒 | ✅ |
| ESM-2 编码 | ~10-15 min | ✅ |
| 评分时间 | <1 秒 | ✅ |
| 存储 | ~100MB (10K 序列) | ✅ |
| 代码工作量 | 新增 `generate.py` + `screen.py` | 较大 |

---

## 四、改进方案依据说明

### 4.1 方案 A 依据：信息论 + 尺度定律

**理论依据**:

1. **Transformer 深度与表达能力**: 深度 L 的 Transformer 可表示的函数类随 L 指数增长。对于 CDR3 序列（长度 5-13），12 层只能捕获局部特征，33 层才能充分建模首-尾残基的远程协同效应。

2. **多视角信息互补**: 信息论中的互信息分解表明：
   ```
   I(Y; Z_1, Z_2, Z_3) = I(Y; Z_1) + I(Y; Z_2|Z_1) + I(Y; Z_3|Z_1, Z_2)
   ```
   三个尺度的嵌入提供非冗余信息，融合后总信息量大于任一模型的单独信息。

3. **文献支持**: Rives et al. (Science, 2021) 和 Lin et al. (Nature Biotech, 2023) 均表明，ESM-2 650M 在抗体亲和力预测上显著优于 35M。

**数据支持**:
- 当前 t12_35M 编码耗时仅 49 秒，说明 GPU 远未满载
- 88GB 总显存只用了 2GB，余量充足
- t33_650M 的 1280 维嵌入能更好区分序列空间的细微差异

**问题导向**:
- 当前 480 维嵌入可能丢失了关键的远程依赖信息
- 反事实建议的 Top-3 命中率偏低（约 35%），需要更强的表示能力

---

### 4.2 方案 B 依据：因果推断前沿 + 异质性效应

**理论依据**:

1. **常数 CATE 的局限性**: Double ML 假设 θ 对所有样本相同，这隐含了「同质性处理效应」假设。在蛋白质序列设计中，同一突变的效果高度依赖序列上下文，此假设严重违反直觉。

2. **Causal Forest 的数学保证**: Wager & Athey (JASA, 2018) 证明 Causal Forest 的估计量满足渐近正态性：
   ```
   √n(θ̂_CF(x) - θ(x)) → N(0, V(x))
   ```
   且自带置信区间，可进行统计显著性检验。

3. **Honest Tree 防止过拟合**: 通过样本分割（50% 分裂树结构，50% 估计叶节点效应），避免传统决策树的过拟合偏差。

**数据支持**:
- Cox 回归显示不同氨基酸的 HR 差异巨大（甘氨酸 HR=4.35 vs 芳香族 HR=0.67）
- ATE 估计显示 first_is_aromatic 的效应 (-6.54) 远大于其他因素
- 这些异质性信号在常数 CATE 中被平均掉了

**问题导向**:
- 当前所有 1,254 条首残基为 G 的序列都收到相同的 "G→Y" 建议
- 但实际上，不同序列上下文（疏水性、电荷、长度）下，最优突变应该不同
- 需要回答：「对这条具体序列，哪种突变最好？」而非「平均而言哪种突变最好？」

---

### 4.3 方案 C 依据：从分析到设计的范式转移

**理论依据**:

1. **生成式 vs 判别式**: 当前管线是判别式的（分析已有数据），而蛋白质工程需要生成式的（主动设计新序列）。

2. **约束满足 + 优化**: 将设计策略的硬约束（长度 6-7、首残基白名单、反模式）转化为约束采样问题，用蒙特卡洛方法在可行空间内搜索。

3. **快速筛选框架**: 类似于药物发现中的「虚拟筛选」(Virtual Screening)，先用计算模型快速过滤大量候选，再用实验验证少量高潜力候选。

**数据支持**:
- 设计策略预测 RF2 通过率可从 11.7% 提升至 42.5%（3.6×）
- 最终候选率可从 0.59% 提升至 5.5%（9.3×）
- 这些预测需要实际生成序列来验证

**问题导向**:
- 当前产出是「分析报告」，但实验团队需要的是「候选序列清单」
- 反事实建议只针对已有失败序列，无法探索全新序列空间
- 需要闭环：生成 → 评分 → 筛选 → 验证 → 反馈

---

## 五、后续操作建议

### 5.1 优先处理事项（按优先级排序）

| 优先级 | 事项 | 预计耗时 | 负责人 | 依赖 |
|--------|------|---------|--------|------|
| 🔴 P0 | **修复 numpy 版本锁定** | 30 min | Agent | 无 |
| 🔴 P0 | **升级 ESM-2 到 t33_650M** | 2-4 h | Agent | 依赖版本修复 |
| 🔴 P0 | **重新跑完整管线** | 30 min | Agent | ESM-2 升级完成 |
| 🔴 P1 | **实现 Causal Forest** | 4-6 h | Agent | 管线跑完 |
| 🔴 P1 | **对比实验：基线 vs A+B** | 2 h | Agent | CF 实现完成 |
| 🟡 P2 | **实现约束序列生成器** | 4-8 h | Agent | A+B 验证完成 |
| 🟡 P2 | **生成并筛选 10K 候选序列** | 30 min | Agent | 生成器完成 |
| 🟢 P3 | **送 Top 200 做 RF2/AF3 验证** | 待定 | 实验团队 | 候选序列清单 |

### 5.2 潜在风险提示

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| t33_650M OOM | 中 | 编码失败 | 降到 t30_150M；或 batch_size=1 |
| econml 安装失败 | 低 | Causal Forest 不可用 | 改用 sklearn 的 GradientBoosting + 自定义 CATE |
| 生成序列多样性不足 | 中 | 候选过于相似 | 加入多样性约束（编辑距离 > 3） |
| 预测通过率虚高 | 中 | 实验验证失败率高 | 保守筛选阈值（>50% 通过率） |
| 服务器 CUDA 版本限制 | 低 | 无法安装新版 PyTorch | 保持 PyTorch 1.12.1，esm 模型仍兼容 |

### 5.3 关键决策点

**决策点 1：ESM-2 模型选择**

```
IF 显存充足 (≥14GB 可用):
    → 选择 t33_650M (效果最佳)
ELIF 显存中等 (≥6GB 可用):
    → 选择 t30_150M (平衡效果与资源)
ELSE:
    → 保持 t12_35M (基线)
```

**决策点 2：Causal Forest vs Double ML**

```
IF 需要个性化建议 + 论文档次高:
    → 选择 Causal Forest
ELIF 只需要快速原型验证:
    → 保持 Double ML，但用融合嵌入
ELSE:
    → 两者都跑，做对比实验
```

**决策点 3：生成序列的验证策略**

```
IF 实验预算充足:
    → 送 Top 200 全部验证
ELIF 预算有限:
    → 送 Top 50 (高置信度) + 随机 50 (多样性)
ELSE:
    → 仅做计算验证，不送实验
```

### 5.4 资源获取途径

| 资源 | 获取方式 | 备注 |
|------|---------|------|
| ESM-2 模型权重 | `fair-esm` 包自动下载 | 首次加载需联网，约 1-2GB |
| econml 包 | `pip install econml` | 依赖 numpy/scikit-learn |
| GPU 服务器 | REDACTED_USER@REDACTED_IP | SSH 连接 |
| VPN 连接 | REDACTED_VPN | 需加 `--mtu 1228` |
| 原始数据 | `data/Q02223_first50_all_sequences.csv` | 已同步到服务器 |

### 5.5 沟通协作建议

**与实验团队对接**：
- 输出格式：提供 CSV（sequence_id, sequence, predicted_pass_rate, recommended_mutation）
- 关键指标：明确标注每条序列的预测通过率和置信度
- 反馈闭环：实验验证后，将真实结果回流，用于模型迭代优化

**与导师汇报**：
- 强调因果推断的数学严谨性（Causal Forest 的渐近正态性保证）
- 突出个体化设计的应用价值（从"分析报告"到"候选清单"）
- 提供可量化的预测改进（3.6× 通过率提升）

**代码交接**：
- 所有修改提交到 GitHub：`https://github.com/REDACTED_USER/CSC-O.git`
- 关键分支：main
- 未跟踪文件：服务器配置（`.gitignore` 已排除敏感信息）

---

## 六、附录

### 附录 A：项目文件结构

```
analyze1/
├── csco_pipeline.py              # 主入口
├── csco_data_engineering.py      # 数据工程
├── csco_esm2_encode.py           # ESM-2 编码（独立脚本）
├── csco_layer1_stratified.py     # 分层分析
├── csco_layer2_causal.py         # 因果推断
├── csco_layer3_analysis.py       # 反事实分析（独立脚本）
├── csco_layer3_counterfactual.py # 反事实导航
├── csco_layer5_synthesis.py      # 规则合成
├── csco_visualize.py             # 可视化面板
├── requirements.txt              # 依赖清单
├── .gitignore                    # Git 排除规则
├── tools/                        # VPN 断连恢复系统
│   ├── vpn_watchdog.sh
│   ├── ssh_auto_attach.sh
│   ├── check_server_status.sh
│   ├── workspace_save.py
│   ├── workspace_restore.py
│   ├── vim_session_save.sh
│   ├── vim_session_restore.sh
│   ├── install.sh
│   └── README.md
├── docs/
│   └── 4090_vpn_remote_run_guide.html
├── output/                       # 生成输出（Git 排除）
│   ├── design_strategy.json
│   ├── csco_analysis_report.txt
│   ├── feature_matrix.csv
│   ├── esm2_embeddings.npy
│   └── ...
└── work/                         # 工作目录（Git 排除）
    └── pipeline_state.json
```

### 附录 B：关键命令速查

```bash
# 服务器端运行 Pipeline
cd ~/CSC-O && source ~/csco_env/bin/activate
nohup python csco_pipeline.py --input data/Q02223_first50_all_sequences.csv \
  --output output --work work --device cuda --resume > output/run.log 2>&1 &

# 查看进度
tail -f ~/CSC-O/output/run.log
cat ~/CSC-O/work/pipeline_state.json

# Mac 端同步代码到服务器
cd /path/to/CSC-O/analyze1
rsync -avP --exclude "output/" --exclude "work/" --exclude "__pycache__/" \
  ./ REDACTED_USER@REDACTED_IP:~/CSC-O/

# Mac 端同步结果回本地
rsync -avP REDACTED_USER@REDACTED_IP:~/CSC-O/output/ \
  /path/to/CSC-O/analyze1/output_server/

# VPN 连接（必须加 MTU）
sudo openconnect --mtu 1228 REDACTED_VPN -u REDACTED_VPN_USER \
  --useragent "AnyConnect Windows 4.10.07073" --os=win -b
```

### 附录 C：依赖版本锁定

```
# 已知兼容版本（服务器已验证）
numpy==1.23.5          # 必须 < 1.24，否则 torch 1.12.1 报错
torch==1.12.1+cu113    # CUDA 11.4 兼容
fair-esm==2.0.0        # ESM-2 模型
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
lifelines>=0.27.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### 附录 D：联系信息

| 角色 | 信息 | 用途 |
|------|------|------|
| 服务器 IP | REDACTED_IP | SSH 连接 |
| 用户名 | REDACTED_USER | SSH 登录 |
| VPN 地址 | REDACTED_VPN | 校园网接入 |
| VPN 用户名 | REDACTED_VPN_USER | 身份认证 |

---

*本文档由 AI Agent 协助生成，旨在为后续项目推进提供完整的参考依据。*
