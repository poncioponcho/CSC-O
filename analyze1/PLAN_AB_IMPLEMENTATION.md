# CSC-O 方案A+B整合实施文档

**实施日期**: 2026-06-02  
**版本**: v2.1  

---

## 1. 方案设计

### 1.1 方案A：分层ATE (Stratified Average Treatment Effect)

**问题**: 全局线性ATE无法捕捉芳香比在不同CDR3长度下的异质性因果效应（长度6保护、长度5风险），导致ATE-Cox矛盾。

**解决方案**: 在每个CDR3长度内独立估计ATE，避免跨长度异质性导致的偏差。

**架构**:

```
_estimate_all_ate()
    ├── 全局ATE (原有, 输出 ate_estimates.csv)
    └── _estimate_stratified_ate() (新增)
         ├── 长度5: aromatic_ratio ATE on PAE = ?
         ├── 长度6: aromatic_ratio ATE on PAE = ?
         ├── 长度7: aromatic_ratio ATE on PAE = ?
         └── ... (所有长度 × 所有处理变量)
         输出: stratified_ate_estimates.csv
```

**分层ATE混杂变量**: 每个长度子集内仅使用 `backbone_id` 作为混杂变量（无需再控制cdr3_len，因为已按长度分层）。

### 1.2 方案B：长度差异化约束 (Length-Differentiated Constraints)

**问题**: 当前策略对所有CDR3长度使用相同的 `soft_preferences`，但不同长度下各氨基酸的因果效应方向和强度不同。

**解决方案**: 在策略JSON中新增 `length_specific_preferences` 字段，按长度覆盖默认 `soft_preferences`。

**约束规则定义**:

| 约束参数 | 规则 | 依据 |
|---------|------|------|
| `aromatic_min_ratio` | ATE>0 → 0.0; ATE<-1.0 → 保持默认; 其他 → 75%默认值 | 分层ATE方向和强度 |
| `glycine_max_ratio` | ATE>2.0 → min(默认, 0.15); 其他 → 保持默认 | 甘氨酸风险强度 |
| `serine_max_ratio` | ATE>2.0 → min(默认, 0.12); 其他 → 保持默认 | 丝氨酸风险强度 |
| `proline_max_count` | ATE<0 → 2 (放宽); 其他 → 1 (保持) | 脯氨酸保护效应 |

### 1.3 整合架构

```
Pipeline执行流程:
    stage_layer2_causal()
        ├── _estimate_all_ate()
        │    ├── 全局ATE → ate_estimates.csv
        │    └── _estimate_stratified_ate() → stratified_ate_estimates.csv
        └── 返回 stratified_results

    stage_layer5_synthesis()
        ├── 读取 stratified_ate_estimates.csv
        ├── 构建 strat_ate_map[(length, treatment)] → ATE
        ├── 按长度自动生成长度差异化约束:
        │    for length in valid_lengths:
        │        根据分层ATE方向/强度 → length_specific_prefs[length]
        └── 写入 design_strategy.json:
             ├── soft_preferences (全局默认)
             ├── length_specific_preferences (按长度覆盖)
             └── length_generation_weights (按通过率加权)

Generator执行流程:
    generate_cdr3(strategy, length, ...)
        ├── lsp = strategy['length_specific_preferences']
        ├── len_prefs = lsp[str(length)]
        ├── effective_sp = {**soft_preferences, **len_prefs}  ← 长度覆盖全局
        └── 使用 effective_sp 进行氨基酸采样
```

---

## 2. 代码变更清单

### 2.1 csco_pipeline.py

| 变更 | 位置 | 说明 |
|------|------|------|
| 新增 `_estimate_stratified_ate()` | ~L620-670 | 按CDR3长度分层估计ATE，输出CSV |
| 修改 `_estimate_all_ate()` 返回值 | ~L613 | 返回 `(ate_results, stratified_results)` |
| 修改 `stage_layer2_causal()` 调用 | ~L487 | 接收双返回值 |
| 新增 `strat_ate_map` 构建 | ~L1120-1130 | 读取分层ATE构建映射 |
| 新增 `length_specific_prefs` 生成 | ~L1132-1170 | 按分层ATE自动生成差异化约束 |
| 新增 `length_specific_preferences` 字段 | ~L1198 | 写入策略JSON |

### 2.2 csco_generator.py

| 变更 | 位置 | 说明 |
|------|------|------|
| 新增 `lsp`/`len_prefs`/`effective_sp` | ~L58-60 | 读取并合并长度特定约束 |
| `sp` → `effective_sp` | ~L85-98 | 所有soft preference引用改为effective_sp |
| 新增 `proline_max_count` 条件权重 | ~L94-95 | proline权重根据max_count动态调整 |

---

## 3. 验证结果

### 3.1 语法检查

| 文件 | 结果 |
|------|------|
| csco_pipeline.py | ✅ 通过 |
| csco_generator.py | ✅ 通过 |

### 3.2 功能测试

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 长度特定偏好合并 | ✅ | Length5: aro=0.0, gly=0.15; Length6: aro=0.2, gly=0.2; Length7: aro=0.15, gly=0.2, pro=2 |
| 序列生成-长度5无芳香偏好 | ✅ | avg aromatic_ratio=0.297 (无min_ratio约束) |
| 序列生成-长度7允许proline=2 | ✅ | max proline_count=2 |
| `_estimate_stratified_ate` 函数存在 | ✅ | 可正常导入 |
| 分层ATE → 约束自动生成链路 | ✅ | strat_ate_map → length_specific_prefs → JSON |

### 3.3 预期运行结果 (基于诊断数据)

当pipeline在服务器上重新运行时，`length_specific_preferences` 将根据分层ATE自动生成：

```json
{
  "length_specific_preferences": {
    "5": {
      "aromatic_min_ratio": 0.0,
      "glycine_max_ratio": 0.15,
      "serine_max_ratio": 0.12,
      "proline_max_count": 1
    },
    "6": {
      "aromatic_min_ratio": 0.2,
      "glycine_max_ratio": 0.15,
      "serine_max_ratio": 0.15,
      "proline_max_count": 1
    },
    "7": {
      "aromatic_min_ratio": 0.15,
      "glycine_max_ratio": 0.15,
      "serine_max_ratio": 0.15,
      "proline_max_count": 2
    }
  }
}
```

**推导依据** (来自诊断脚本PART 6):
- 长度5: aromatic ATE=+10.7(风险) → aro_min=0.0; glycine ATE>2 → gly_max=0.15
- 长度6: aromatic ATE=-1.9(保护) → aro_min=0.2(保持); glycine ATE>2 → gly_max=0.15
- 长度7: aromatic ATE=-0.5(弱保护) → aro_min=0.15(75%); proline ATE<0 → pro_max=2

---

## 4. 部署步骤

### 4.1 上传新代码

```bash
# 本地Mac终端
rsync -avz --exclude='output_server' --exclude='output_server_v2' \
  /path/to/CSC-O/analyze1/ REDACTED_USER@REDACTED_IP:~/CSC-O/analyze1/
```

### 4.2 运行Pipeline

```bash
# 服务器端
ssh REDACTED_USER@REDACTED_IP
conda activate csco

nohup python3 -u csco_pipeline.py \
    -i ~/CSC-O/Q02223_first50_all_sequences.csv \
    -o ~/CSC-O/output_v4 -w ~/CSC-O/work_v4 -d cuda \
    --esm2-models esm2_t12_35M_UR50D,esm2_t30_150M_UR50D \
    --esm2-fusion concat --cate-method r_learner \
    --subgroup-clusters 3 --target final_candidate \
    > ~/CSC-O/run_v4.log 2>&1 &
```

### 4.3 验证输出

```bash
# 检查分层ATE输出
cat ~/CSC-O/output_v4/stratified_ate_estimates.csv | head -20

# 检查长度差异化约束
cat ~/CSC-O/output_v4/design_strategy.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('length_specific_preferences:')
for k, v in d.get('length_specific_preferences', {}).items():
    print(f'  Length {k}: {v}')
"
```

---

## 5. 回滚机制

如果新方案出现问题，可通过以下方式回滚：

1. **策略层面**: 删除 `length_specific_preferences` 字段，generator自动回退到全局 `soft_preferences`
2. **代码层面**: `git revert` 回滚到v2.0版本
3. **运行层面**: 使用 `--target rf2_passed` 恢复旧行为
