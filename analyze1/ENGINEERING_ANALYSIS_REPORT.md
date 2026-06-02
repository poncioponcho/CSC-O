# CSC-O 工程分析报告：ATE-Cox矛盾诊断与策略优化

**项目**: CSC-O (Causal-Stratified Counterfactual Optimization)  
**分析日期**: 2026-06-02  
**分析人员**: 自动化工程分析  
**文档版本**: 1.0  

---

## 1. 数据获取过程

### 1.1 云端Pipeline执行

| 项目 | 详情 |
|------|------|
| 服务器 | REDACTED_USER@REDACTED_IP (REDACTED_HOSTNAME) |
| Conda环境 | csco |
| 执行命令 | `nohup python3 -u csco_pipeline.py -i ~/CSC-O/Q02223_first50_all_sequences.csv -o ~/CSC-O/output_v3 -w ~/CSC-O/work_v3 -d cuda --esm2-models esm2_t12_35M_UR50D,esm2_t30_150M_UR50D --esm2-fusion concat --cate-method r_learner --subgroup-clusters 3 --target final_candidate > ~/CSC-O/run_v3.log 2>&1 &` |
| PID | 2856 |
| 完成时间 | 2026-06-02 |

### 1.2 各阶段运行耗时

| 阶段 | 耗时 | 说明 |
|------|------|------|
| data_engineering | ~2s | 特征矩阵构建 |
| layer1_stratified | 1.8s | 分层归因+生存分析 |
| layer2_causal | 4.3s | 因果DAG+ATE估计 |
| layer3_esm2_encode | 1.6s | ESM-2嵌入(已缓存, 10572×1120) |
| layer3_counterfactual | **2894.6s (48min)** | R-learner CATE估计(主要瓶颈) |
| layer5_synthesis | 0.1s | 策略合成 |

### 1.3 数据下载

| 操作 | 命令 | 结果 |
|------|------|------|
| 代码上传 | `rsync -avz ... REDACTED_USER@REDACTED_IP:~/CSC-O/analyze1/` | 81文件, 179997 bytes |
| 结果下载 | `rsync -avz REDACTED_USER@REDACTED_IP:~/CSC-O/output_v3/ ... /output_v3_server/` | PNG+JSON+TXT |
| CSV数据下载 | rsync (需密码交互, sandbox不支持) | **失败** — 使用服务器端诊断脚本替代 |

### 1.4 数据完整性验证

| 文件 | v1 (output_server) | v3 (output_v3_server) | 状态 |
|------|-------------------|----------------------|------|
| design_strategy.json | ✅ version=1.0 | ✅ version=2.0 | 已验证 |
| csco_analysis_report.txt | ✅ | ✅ | 已验证 |
| design_strategy.txt | ✅ | ✅ | 已验证 |
| ate_estimates.csv | ❌ 未下载 | ❌ 未下载 | 从服务器输出获取 |
| cox_hazard_ratios.csv | ❌ 未下载 | ❌ 未下载 | 从诊断脚本获取 |
| diagnostic_aromatic.py | N/A | ✅ 服务器运行成功 | 7模块全部完成 |
| diagnostic_report.txt | N/A | ✅ 完整输出 | 已通过tee保存 |

**替代方案**: 由于CSV文件无法通过sandbox下载, 在服务器端运行 `diagnostic_aromatic.py` 完成所有统计分析, 输出通过 `tee` 保存到 `~/CSC-O/diagnostic_report.txt`。

---

## 2. v1 → v2.0 策略对比

### 2.1 硬约束变更

| 约束 | v1 | v2.0 | 变更原因 |
|------|-----|------|---------|
| `last_residue_whitelist` | ['A','S','V','Y'] | **['A','S','V']** | Y的ATE=+1.30(风险), 通过率仅10.9% |
| 首残基白名单阈值 | >10% | **>15%** | 提高筛选标准 |
| `optimization_target` | 无(默认rf2_passed) | **final_candidate** | 对齐最终目标 |
| `length_generation_weights` | 无 | **{5:0.18, 6:0.2261, 7:0.516}** | 按通过率加权 |

### 2.2 ATE估计变更 (关键!)

| 变量 | v1 ATE on PAE | v2.0 ATE on PAE | 变化 | 原因 |
|------|--------------|-----------------|------|------|
| **glycine_ratio** | **-0.94** | **+5.41** | **+6.35 翻转** | 加入cdr3_len混杂解决甘氨酸悖论 |
| **aromatic_ratio** | **-1.02** | **+5.03** | **+6.05 翻转** | 加入cdr3_len混杂,但引发新矛盾(见§3) |
| **serine_ratio** | **+9.69** | **+0.40** | **-9.29 剧减** | 长度混杂吸收了大部分丝氨酸效应 |
| **proline_count** | +1.35 | -0.08 | -1.43 翻转 | 长度混杂修正 |
| first_is_aromatic | -6.54 | -4.69 | +1.85 | 仍为强保护因子 |
| last_is_YH | +2.88 | +1.30 | -1.58 | 仍为风险因子 |

### 2.3 Cox HR (未变)

| 变量 | HR | 含义 |
|------|-----|------|
| glycine_ratio | 4.35 | 最强风险因子 |
| serine_ratio | 1.77 | 风险因子 |
| hydrophobic_ratio | 1.18 | 弱风险 |
| cdr3_len | 1.08 | 弱风险 |
| positive_ratio | 0.95 | 中性 |
| **aromatic_ratio** | **0.67** | **保护因子** |

---

## 3. 芳香比ATE-Cox矛盾深度诊断

### 3.1 问题描述

| 模型 | aromatic_ratio效应 | 方向 |
|------|-------------------|------|
| ATE (v2, backbone+cdr3_len) | +5.03 on PAE | 风险(增加PAE) |
| ATE (完全混杂, +全部ratio) | +8.02 on PAE | 更强风险 |
| Cox PH (控制全部ratio) | HR=0.67 | 保护(降低失败风险) |

### 3.2 诊断方法

运行 `diagnostic_aromatic.py` (7模块), 服务器端执行:
```bash
python3 -u diagnostic_aromatic.py 2>&1 | tee ~/CSC-O/diagnostic_report.txt
```

### 3.3 PART 1: 共线性分析

**强相关对 (|r|>=0.3):**

| 变量对 | Pearson r | 含义 |
|--------|----------|------|
| cdr3_len ↔ serine_ratio | **+0.547** | 长CDR3含更多丝氨酸 |
| aromatic_ratio ↔ hydrophobic_ratio | -0.357 | 芳香与疏水残基互斥 |
| glycine_ratio ↔ serine_ratio | -0.322 | 甘氨酸与丝氨酸互斥 |

**关键发现**: `serine_ratio` 与 `cdr3_len` 的强相关(r=+0.547)解释了丝氨酸ATE从+9.69暴跌到+0.40——之前的高风险几乎全部来自长度混杂。

### 3.4 PART 2: 渐进式ATE (核心证据)

| 混杂变量集 | aromatic_ratio ATE | 变化量 |
|-----------|-------------------|--------|
| 无混杂 | **-1.019** (保护) | — |
| + backbone_id | -1.019 | -0.001 |
| **+ cdr3_len** | **+5.032** (风险) | **+6.052 翻转!** |
| + glycine_ratio | +6.096 | +1.064 |
| + serine_ratio | +6.877 | +0.781 |
| + 全部ratio | **+8.018** (更强风险) | +1.141 |

**关键发现**:
1. `backbone_id` 几乎无影响 (delta=-0.001)
2. **加入 `cdr3_len` 是ATE翻转的直接原因** (delta=+6.052)
3. 加入更多ratio混杂后ATE反而更大 (+8.018), 说明不是缺失混杂问题
4. 矛盾**不随混杂变量增加而缓解**, 反而加剧

### 3.5 PART 3: 全变量渐进ATE对比

| 处理变量 | Raw ATE | Min ATE | Full ATE | 符号翻转? |
|---------|---------|---------|----------|----------|
| aromatic_ratio | -1.019 | +5.032 | **+8.018** | — |
| glycine_ratio | -0.937 | +5.412 | **+2.922** | — |
| serine_ratio | +9.682 | +0.396 | +0.736 | — |
| proline_count | +1.346 | -0.081 | +0.026 | **YES** |
| first_is_aromatic | -6.531 | -4.687 | -4.616 | — |
| last_is_YH | +2.882 | +1.297 | +1.906 | — |

**注意**: glycine_ratio的Full ATE(+2.922)比Min ATE(+5.412)低, 说明加入其他ratio混杂后甘氨酸效应被部分吸收。

### 3.6 PART 4: Cox回归复现

| 变量 | Coef | HR | 95% CI | p |
|------|------|-----|--------|---|
| cdr3_len | +0.0779 | 1.081 | [1.071, 1.091] | 2.09e-64*** |
| positive_ratio | -0.0483 | 0.953 | [0.731, 1.241] | 7.21e-01 |
| **aromatic_ratio** | **-0.4049** | **0.667** | **[0.522, 0.852]** | **1.17e-03** |
| glycine_ratio | +1.4709 | 4.353 | [3.464, 5.471] | 1.83e-36*** |
| serine_ratio | +0.5701 | 1.768 | [1.461, 2.141] | 4.90e-09*** |
| hydrophobic_ratio | +0.1680 | 1.183 | [0.986, 1.420] | 7.12e-02 |

Concordance: 0.7999

### 3.7 PART 5: Backbone_id影响

| 混杂 | ATE | Delta |
|------|-----|-------|
| 无 | -1.019 | — |
| + backbone_id | -1.019 | -0.001 |
| + backbone_id + len | +5.032 | +6.052 |
| + 全部ratio | +8.018 | +2.986 |

**结论**: backbone_id不是矛盾来源, cdr3_len才是。

### 3.8 PART 6: 分层分析 (决定性证据)

| CDR3长度 | N | aromatic ATE on PAE | 方向 | 显著性 |
|----------|---|---------------------|------|--------|
| **5** | 1200 | **+10.699** | ❌ 强风险 | *** |
| **6** | 1132 | **-1.911** | ✅ 保护 | * |
| **7** | 1188 | **-0.477** | 中性 | n.s. |
| 8 | 1192 | +1.956 | 风险 | ** |
| 9 | 1108 | +10.934 | 强风险 | *** |
| 10 | 1176 | +21.331 | 极强风险 | *** |
| 11 | 1300 | +18.413 | 极强风险 | *** |
| 12 | 1164 | +14.299 | 强风险 | *** |
| 13 | 1112 | -6.300 | 保护 | *** |

**决定性发现**: 芳香比的因果效应在不同CDR3长度下**方向完全相反**:
- 长度6: 保护 (ATE=-1.911) → 与Cox HR=0.67一致 ✅
- 长度7: 中性 (ATE=-0.477) → 与Cox基本一致 ✅
- 长度5: 强风险 (ATE=+10.699) → 与Cox矛盾 ❌
- 长度8-12: 强风险 (ATE=+10~+21) → 与Cox矛盾 ❌

---

## 4. 问题根源分析

### 4.1 根因1: 组合数据(Compositional Data)问题

CDR3氨基酸ratio是组合数据——各ratio之和 ≤ 1。这意味着:
- ratio变量不独立: 芳香比升高时, 其他ratio必然下降
- 控制其他ratio是逻辑矛盾: "芳香比升高而甘氨酸比/丝氨酸比不变"在物理上不可能
- Cox和ATE都问了一个不可能的问题, 但模型假设不同导致答案不同

### 4.2 根因2: 异质性因果效应

芳香比的因果效应在不同CDR3长度下方向相反。全局线性ATE把所有长度的效应加权平均, 得到一个无意义的中间值(+5.03)。

**机制解释**:
- **短CDR3(5)**: 芳香残基占比过高(>20%)导致空间位阻, 破坏loop构象 → 风险
- **中等CDR3(6-7)**: 芳香残基通过π-堆积稳定loop-抗原界面 → 保护
- **长CDR3(8-12)**: 芳香残基聚集导致loop刚性过强, 无法形成正确构象 → 风险
- **极长CDR3(13)**: 特殊构象使得芳香残基重新发挥稳定作用 → 保护

### 4.3 根因3: 线性混杂变量假设失效

控制 `cdr3_len` 作为**线性**混杂变量无法捕捉芳香比与长度之间的**非线性交互效应**。当效应在长度6为负、长度5/8-12为正时, 线性控制会产生Simpson悖论式的翻转。

---

## 5. 技术解决方案

### 5.1 方案A: 按长度分层ATE估计 (推荐)

**原理**: 在每个CDR3长度内独立估计ATE, 避免跨长度异质性导致的偏差。

**实施**:
```python
# 替代当前的全局ATE
for length in [5, 6, 7]:
    sub = feat_df[feat_df['cdr3_len'] == length]
    ate, se, t, p, ci_l, ci_u = backdoor_ate(sub, treatment, outcome, ['backbone_id'])
```

**预期结果**:
- 长度6: aromatic ATE ≈ -1.9 (保护) → 保留 aromatic_min_ratio
- 长度7: aromatic ATE ≈ -0.5 (中性) → 可降低 aromatic_min_ratio
- 长度5: aromatic ATE ≈ +10.7 (风险) → 移除 aromatic_min_ratio

### 5.2 方案B: 长度差异化策略约束

```python
length_specific_preferences = {
    5: {'aromatic_min_ratio': 0.0, 'glycine_max_ratio': 0.15},
    6: {'aromatic_min_ratio': 0.2, 'glycine_max_ratio': 0.2},
    7: {'aromatic_min_ratio': 0.15, 'glycine_max_ratio': 0.2},
}
```

### 5.3 方案C: 交互项ATE模型

在ATE估计中加入 `aromatic_ratio × cdr3_len` 交互项:
```python
feat_df['aromatic_x_len'] = feat_df['aromatic_ratio'] * feat_df['cdr3_len']
confounders = ['backbone_id', 'cdr3_len', 'aromatic_x_len']
```

### 5.4 方案优先级

| 方案 | 预期效果 | 实施难度 | 推荐度 |
|------|---------|---------|--------|
| A: 分层ATE | 精确捕捉异质性 | 低 | ⭐⭐⭐⭐⭐ |
| B: 长度差异化约束 | 策略更精准 | 低 | ⭐⭐⭐⭐ |
| C: 交互项模型 | 数学上更优雅 | 中 | ⭐⭐⭐ |

**建议**: 先实施方案A+B, 验证效果后再考虑方案C。

---

## 6. 其他关键发现

### 6.1 甘氨酸悖论已解决

| 版本 | glycine_ratio ATE | 混杂变量 |
|------|-------------------|---------|
| v1 | -0.94 (看似保护) | backbone_id only |
| v2 | +5.41 (真实风险) | backbone_id + cdr3_len |
| Full | +2.92 (风险) | backbone_id + cdr3_len + 全部ratio |
| Cox | HR=4.35 (强风险) | 全部ratio |

加入cdr3_len后甘氨酸从"保护"翻转为"风险", 与Cox一致。Full ATE(+2.92)低于Min ATE(+5.41)说明部分效应被其他ratio吸收, 但方向始终为正(风险)。

**策略影响**: `glycine_max_ratio=0.2` 方向正确, 可考虑降至0.15。

### 6.2 丝氨酸效应被严重高估

| 版本 | serine_ratio ATE | 说明 |
|------|-------------------|------|
| v1 | +9.69 | 极强风险(与cdr3_len混杂) |
| v2 | +0.40 | 几乎中性(控制长度后) |
| Full | +0.74 | 中性 |
| Cox | HR=1.77 | 风险 |

丝氨酸与CDR3长度强相关(r=+0.547), 之前的高风险几乎全部来自长度混杂。

**策略影响**: `serine_max_ratio=0.15` 可能过严, 可放宽至0.25。

### 6.3 脯氨酸效应被高估

| 版本 | proline_count ATE |
|------|-------------------|
| v1 | +1.35 (风险) |
| v2 | -0.08 (中性) |
| Full | +0.03 (中性) |

**策略影响**: `proline_max_count=1` 可能过严, 可放宽至2。

---

## 7. 可复现操作步骤

### 7.1 Pipeline运行 (v2.0)

```bash
# 服务器端
ssh REDACTED_USER@REDACTED_IP
conda activate csco

# 上传代码 (本地Mac)
rsync -avz --exclude='output_server' --exclude='output_server_v2' \
  /path/to/CSC-O/analyze1/ REDACTED_USER@REDACTED_IP:~/CSC-O/analyze1/

# 运行pipeline
cd ~/CSC-O/analyze1
nohup python3 -u csco_pipeline.py \
    -i ~/CSC-O/Q02223_first50_all_sequences.csv \
    -o ~/CSC-O/output_v3 -w ~/CSC-O/work_v3 -d cuda \
    --esm2-models esm2_t12_35M_UR50D,esm2_t30_150M_UR50D \
    --esm2-fusion concat --cate-method r_learner \
    --subgroup-clusters 3 --target final_candidate \
    > ~/CSC-O/run_v3.log 2>&1 &

# 监控
tail -f ~/CSC-O/run_v3.log
```

### 7.2 诊断脚本运行

```bash
# 上传诊断脚本 (本地Mac)
rsync -avz diagnostic_aromatic.py REDACTED_USER@REDACTED_IP:~/CSC-O/analyze1/

# 服务器端运行
cd ~/CSC-O/analyze1
python3 -u diagnostic_aromatic.py 2>&1 | tee ~/CSC-O/diagnostic_report.txt
```

### 7.3 结果下载

```bash
# 本地Mac
rsync -avz REDACTED_USER@REDACTED_IP:~/CSC-O/output_v3/ \
  /path/to/CSC-O/analyze1/output_v3_server/
```

---

## 8. 待办事项

- [ ] 实施方案A: pipeline ATE改为按长度分层估计
- [ ] 实施方案B: 长度差异化策略约束
- [ ] 重新运行pipeline验证分层ATE效果
- [ ] 下载完整CSV数据到本地进行更深入分析
- [ ] 更新REFACTORING_REPORT.md中的ATE对比数据

---

## 附录A: 诊断脚本输出完整记录

### A.1 PART 1: 相关矩阵

```
                 cdr3_len  positive_ratio  aromatic_ratio  glycine_ratio  serine_ratio  hydrophobic_ratio
            cdr3_len   +1.000   -0.018   -0.231   -0.240   +0.547*  +0.051
      positive_ratio   -0.018   +1.000   -0.000   -0.109   -0.202   -0.213
      aromatic_ratio   -0.231   -0.000   +1.000   -0.102   -0.273   -0.357
       glycine_ratio   -0.240   -0.109   -0.102   +1.000   -0.322   -0.050
        serine_ratio   +0.547*  -0.202   -0.273   -0.322   +1.000   -0.044
   hydrophobic_ratio   +0.051   -0.213   -0.357   -0.050   -0.044   +1.000
```

### A.2 PART 2: 渐进ATE

```
Confounders                                        ATE       SE        t          p                 95% CI      Delta
  No confounders (raw)                          -1.019    0.221    -4.60   4.21e-06*** [  -1.452,   -0.585]
  + backbone_id (v1)                            -1.019    0.330    -3.09   2.00e-03 ** [  -1.666,   -0.373]     -0.001
  + cdr3_len (v2 current)                       +5.032    0.364   +13.83   0.00e+00*** [  +4.319,   +5.745]     +6.052
  + glycine_ratio                               +6.096    0.371   +16.44   0.00e+00*** [  +5.370,   +6.823]     +1.064
  + serine_ratio                                +6.877    0.383   +17.98   0.00e+00*** [  +6.127,   +7.627]     +0.781
  + hydro+positive (full Cox match)             +8.018    0.384   +20.90   0.00e+00*** [  +7.266,   +8.770]     +1.141
```

### A.3 PART 6: 分层分析

```
 Len      N  aro_mean  PAE_mean  pass_rt       ATE          p
     5   1200    0.2350     10.75    0.180   +10.699***   0.00e+00
     6   1132    0.1737      9.64    0.226    -1.911  *   2.28e-02
     7   1188    0.2374      7.70    0.516    -0.477      5.02e-01
     8   1192    0.2325     14.98    0.008    +1.956 **   2.00e-03
     9   1108    0.1843     13.99    0.017   +10.934***   0.00e+00
    10   1176    0.1875     13.89    0.015   +21.331***   0.00e+00
    11   1300    0.1597     14.24    0.027   +18.413***   0.00e+00
    12   1164    0.1725     15.38    0.023   +14.299***   0.00e+00
    13   1112    0.1627     17.21    0.038    -6.300***   4.93e-10
```
