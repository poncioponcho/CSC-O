# 方案 A+B 数学原理详解：从"平均优化"到"个体化设计"

> **目标读者**: 需要理解数学工具改进原理的研究者  
> **核心诉求**: 体现数学工具对"生成新蛋白质序列的可行优化方向"的改进效果  
> **对比基线**: 当前 CSC-O v1.0 (t12_35M + Double ML 常数 CATE)

---

## 一、问题本质：当前方案的数学局限

### 1.1 当前方案的数学模型

当前 CSC-O v1.0 的核心数学框架：

**表示层**:
```
vh_sequence  →  ESM-2_t12_35M  →  z_i ∈ ℝ^480
```

**因果推断层** (Double ML):
```
Y_i = θ·T_i + f(X_i) + ε_i    (θ 是标量，对所有样本相同)
```

其中:
- `Y_i`: 结果 (RF2 passed 或 PAE)
- `T_i`: 处置 (如 first_is_aromatic ∈ {0,1})
- `X_i`: 协变量 (ESM-2 嵌入 + 序列特征)
- `θ`: **常数 CATE** (Conditional Average Treatment Effect)

**关键局限**:
1. **θ 是标量**: 意味着"所有序列对同一突变的响应完全相同"
2. **嵌入维度 480**: 表达能力有限，可能丢失关键的序列上下文信息
3. **无亚群区分**: 无法回答"哪类序列更适合做哪种突变"

### 1.2 用一个具体例子说明局限

**场景**: 考虑挽救编辑 `Pos0 G→Y`

当前方案的推断:
```
ATE(G→Y) = -1.85  (平均 PAE 降低 1.85)
→ 对所有 1,254 条含 G 首残基的序列，都推荐 G→Y
```

**但现实中**:
- 对序列 A: `GSGTDTQSFTH` → G→Y 可能效果很好 (PAE ↓ 3.0)
- 对序列 B: `GSGTSTEDFKY` → G→Y 可能效果一般 (PAE ↓ 0.5)
- 对序列 C: `GSGTAVSDFTY` → G→Y 可能甚至有害 (PAE ↑ 0.3)

**为什么效果不同？** 因为序列的**整体上下文**（其余位置的氨基酸组合）决定了突变效果。

---

## 二、方案 A：ESM-2 升级的数学原理

### 2.1 Transformer 表达能力的尺度定律

ESM-2 的架构遵循标准的 Transformer Encoder:

```
H^(0) = Embedding(x) + PositionalEncoding
H^(l) = TransformerBlock(H^(l-1)),  l = 1,...,L
z = MeanPool(H^(L))  ∈ ℝ^d
```

| 模型 | 层数 L | 维度 d | 参数量 | 注意力头数 |
|------|--------|--------|--------|-----------|
| t12_35M | 12 | 480 | 35M | 20 |
| t30_150M | 30 | 640 | 150M | 20 |
| **t33_650M** | **33** | **1280** | **650M** | **20** |

**数学原理 1：深度与表达能力**

Transformer 的表达能力随深度 L 指数增长（以 ReLU 激活为例）:

```
# 浅层网络 (L=12) 可表示的函数类
F_12 ⊂ F_30 ⊂ F_33

# 对于长度为 n 的序列，注意力机制可捕获的依赖范围
 receptive_field_size ∝ L × (attention_head_diversity)
```

对于抗体 CDR3（长度 5-13），12 层足够捕获局部特征，但**远程依赖**（如首残基与尾残基的协同效应）需要更深层的非线性变换才能充分表达。

**数学原理 2：宽度与信息容量**

嵌入维度 d 决定了每个位置向量的信息容量:
```
信息容量 ∝ d × log(d)   (基于随机投影理论)

d=480:  可区分 ~2^480 种不同的局部序列模式
d=1280: 可区分 ~2^1280 种模式  (指数级增长)
```

抗体 CDR3 虽然只有 5-13 个残基，但每个位置有 20 种可能，总序列空间是 20^13 ≈ 8×10^16。480 维只能粗略区分大类，而 1280 维可以捕获更精细的序列上下文。

### 2.2 多尺度嵌入融合的信息论视角

三种模型提取的嵌入具有**互补信息**:

| 尺度 | 捕获的信息 | 局限 |
|------|-----------|------|
| t12_35M (480d) | 局部氨基酸性质、疏水性、电荷 | 忽略长程相互作用 |
| t30_150M (640d) | 短程结构 motif、β-turn、loop 模式 | 对全局构象敏感但精度有限 |
| t33_650M (1280d) | 全局折叠模式、抗原结合面特征 | 可能过拟合局部噪声 |

**信息论解释**:

设三种嵌入分别为 Z_1, Z_2, Z_3，下游任务为预测 Y (RF2 pass)。

互信息分解:
```
I(Y; Z_1, Z_2, Z_3) = I(Y; Z_1) + I(Y; Z_2 | Z_1) + I(Y; Z_3 | Z_1, Z_2)
```

由于不同尺度捕获不同层次的信息:
- `I(Y; Z_2 | Z_1) > 0`: t30 提供了 t12 没有的互补信息
- `I(Y; Z_3 | Z_1, Z_2) > 0`: t33 提供了前两者的互补信息

**融合策略**:

```python
# 拼接融合 (最简单有效)
Z_fused = [Z_12 | Z_30 | Z_33] ∈ ℝ^(480+640+1280) = ℝ^2400

# 或加权融合 (学习最优权重)
Z_fused = α·Z_12 + β·Z_30 + γ·Z_33
# α, β, γ 可通过验证集网格搜索确定
```

### 2.3 消融实验设计 (验证改进效果)

```
实验设置:
  基线: 仅用 Z_12 (480d) + Double ML
  对照1: 仅用 Z_33 (1280d) + Double ML
  对照2: Z_12 + Z_30 (1120d) + Double ML
  实验组: Z_12 + Z_30 + Z_33 (2400d) + Double ML

评估指标:
  1. CATE 估计的 MSE (与真实效应对比)
  2. RF2 pass 预测的 AUC
  3. 反事实建议的 Top-3 命中率

假设:
  H1: AUC(Z_33) > AUC(Z_12), p < 0.05
  H2: AUC(融合) > max(AUC(Z_12), AUC(Z_30), AUC(Z_33)), p < 0.05
```

---

## 三、方案 B：Causal Forest 的数学原理

### 3.1 为什么常数 CATE 不够？

当前 Double ML 的数学模型:

```
Y_i = θ·T_i + β^T·X_i + ε_i        (θ ∈ ℝ, 标量)
```

这隐含了一个**强假设**: 处置效应 θ 对所有人相同（Homogeneous Treatment Effect）。

但在蛋白质序列设计中，这个假设严重违反直觉:
- 同样的 G→Y 突变，对疏水性序列可能改善结合（芳香族增加疏水相互作用）
- 但对已富含芳香族的序列可能破坏平衡（疏水性过强导致聚集）

**正确的模型应该是**:

```
Y_i = θ(X_i)·T_i + f(X_i) + ε_i    (θ: ℝ^d → ℝ, 函数)
```

其中 θ(X_i) 是**异质性 CATE**，取决于序列的嵌入表示 X_i。

### 3.2 Causal Forest 的核心数学

Causal Forest 是广义随机森林 (Generalized Random Forest) 在因果推断中的特例。

**核心思想**: 用决策树对特征空间进行自适应划分，使得同一叶节点内的样本具有相似的处置效应。

**分裂准则** (Honest Causal Tree):

不同于标准决策树的 MSE 分裂，因果树最大化**处置效应的异质性**:

```
分裂前:
  Δ_parent = Var(τ | X ∈ parent_node)

分裂后 (左子节点 L, 右子节点 R):
  Δ_split = Var(τ | X ∈ L)·n_L + Var(τ | X ∈ R)·n_R

最优分裂: max(Δ_split - Δ_parent)
```

其中 τ 是局部处置效应:
```
τ(node) = E[Y|T=1, X∈node] - E[Y|T=0, X∈node]
```

**Honesty 原则**:

为了防止过拟合，Causal Forest 使用**样本分割**:
- 50% 样本用于**树结构分裂** (splitting sample)
- 50% 样本用于**叶节点效应估计** (estimation sample)

数学保证:
```
√n(θ̂_CF(x) - θ(x)) → N(0, V(x))   (渐近正态)
```

这意味着 Causal Forest 的估计不仅是个性化的，还**自带置信区间**！

### 3.3 在序列设计中的具体意义

**当前 (常数 CATE)**:
```python
# 对所有序列输出相同的建议
for seq in failed_sequences:
    suggestion = "Pos0 G→Y"  # 因为 ATE = -1.85
```

**升级后 (Causal Forest)**:
```python
# 对每个序列输出个性化建议
for seq in failed_sequences:
    X = esm2_embed(seq)
    
    # 估计该序列对 G→Y 的个性化效应
    cate_g_to_y = causal_forest.predict(X, treatment="G→Y")
    # 输出: 均值 + 置信区间, e.g., -0.3 ± 1.2 (不显著)
    
    # 估计该序列对 G→F 的个性化效应
    cate_g_to_f = causal_forest.predict(X, treatment="G→F")
    # 输出: -2.5 ± 0.8 (显著)
    
    # 最终推荐对该序列最有效的突变
    if cate_g_to_f < cate_g_to_y and cate_g_to_f < -1.96*se_g_to_f:
        suggestion = "Pos0 G→F"  # 对这条序列，F 比 Y 更好！
    else:
        suggestion = "Pos0 G→Y"
```

### 3.4 可验证的假设

**H2: Causal Forest 能发现亚群特异性规则**

```
验证方法:
  1. 用 Causal Forest 估计每条失败序列对每个可能突变的 CATE
  2. 对 CATE 进行聚类 (K-means / 层次聚类)
  3. 检查不同亚群的推荐是否不同

预期结果:
  亚群 A (首残基为 G, 富含疏水残基):
    → 推荐 G→Y (芳香族增加疏水相互作用)
    
  亚群 B (首残基为 G, 已富含芳香族):
    → 推荐 G→V (避免疏水性过强)
    
  亚群 C (首残基为 G, 带正电残基多):
    → 推荐 G→D (引入负电荷平衡)
```

---

## 四、A+B 协同：从"平均规则"到"个体化设计"

### 4.1 协同效应的数学框架

```
输入: 失败序列 seq_i
      ↓
[方案 A] 多尺度 ESM-2 编码
  Z_12 ∈ ℝ^480   (局部性质)
  Z_30 ∈ ℝ^640   (短程 motif)
  Z_33 ∈ ℝ^1280  (全局构象)
      ↓
  Z_fused = [Z_12 | Z_30 | Z_33] ∈ ℝ^2400
      ↓
[方案 B] Causal Forest 个性化效应估计
  对每个候选突变 m ∈ {G→Y, G→F, G→W, G→V, ...}:
    τ_m(Z_fused) = CausalForest.predict(Z_fused, treatment=m)
    SE_m = CausalForest.std(Z_fused, treatment=m)
      ↓
  筛选: 保留 τ_m < -1.96·SE_m 的突变 (95% 置信度下显著有效)
      ↓
  排序: 按 τ_m 从小到大排序 (PAE 降低越多越优先)
      ↓
输出: 针对 seq_i 的 Top-3 个性化挽救建议
```

### 4.2 对比：基线 vs A+B

| 场景 | 基线 (t12 + Double ML) | A+B (融合 + Causal Forest) |
|------|----------------------|---------------------------|
| 序列 A: `GSGTDTQSFTH` | G→Y (ATE=-1.85) | **G→Y** (CATE=-2.8±0.4) ✅ |
| 序列 B: `GSGTSTEDFKY` | G→Y (ATE=-1.85) | **G→F** (CATE=-2.1±0.5) ⚡ |
| 序列 C: `GSGTAVSDFTY` | G→Y (ATE=-1.85) | **暂不推荐** (CATE=-0.3±1.1) ❌ |

**关键差异**:
- 基线对所有序列输出相同的 Top-1 建议 (G→Y)
- A+B 对每条序列输出**个性化**的 Top-3 建议，并标注**置信度**

### 4.3 对"生成新序列"的改进效果

**当前生成逻辑**:
```python
def generate_new_sequence():
    # 随机采样，然后用硬约束过滤
    seq = random_cdr3(length=7)
    if first_residue in ['F','V','W','Y'] and 'GGG' not in seq:
        return seq
```

**问题**: 只知道"什么不好"，不知道"对这条序列什么最好"

**A+B 升级后的生成逻辑**:
```python
def generate_new_sequence_v2():
    # 1. 采样候选序列
    candidates = [random_cdr3(length=7) for _ in range(1000)]
    
    # 2. 编码
    Z_fused = [esm2_fusion_embed(seq) for seq in candidates]
    
    # 3. 预测每条候选序列的"可优化潜力"
    # 即：如果对这条序列做最优突变，PAE 能降低多少？
    potential = []
    for z in Z_fused:
        best_cate = min(causal_forest.predict(z, m) for m in all_mutations)
        potential.append(best_cate)
    
    # 4. 选择"可优化潜力最大"的序列作为种子
    # 这些序列只需要一个精准突变就能从失败变为成功
    best_seed = candidates[np.argmin(potential)]
    
    # 5. 对该种子应用个性化最优突变
    best_mutation = argmin_m(causal_forest.predict(Z_fused[best_seed], m))
    
    return apply_mutation(best_seed, best_mutation)
```

**改进效果**:
- 从"随机生成 + 硬约束过滤"升级为"定向生成 + 个性化优化"
- 生成的序列不仅满足约束，而且**被预测为只需最小突变即可成功**

---

## 五、可验证的假设与实验设计

### 假设 H1: 多尺度嵌入提升预测精度

```
零假设 H0: AUC(t33) ≤ AUC(t12)
备择假设 H1: AUC(t33) > AUC(t12)

检验方法:
  1. 用 5-fold 交叉验证训练 RF2 pass 预测器
  2. 分别用 Z_12, Z_33, Z_fused 作为输入
  3. 计算各模型的 AUC
  4. 用 DeLong 检验比较 AUC 差异

样本量: n=10572
检验功效: 若真实差异 ΔAUC=0.05, α=0.05, 功效 > 99%
```

### 假设 H2: Causal Forest 发现更多亚群特异性规则

```
零假设 H0: 常数 CATE 与异质性 CATE 的预测误差相同
备择假设 H1: Causal Forest 的 CATE MSE < Double ML 的 CATE MSE

检验方法:
  1. 将数据分为训练集 (80%) 和测试集 (20%)
  2. 训练集上估计 CATE
  3. 测试集上计算 CATE 的均方误差 (MSE)
     MSE = E[(τ̂(X) - τ_true)^2]
  4. 比较 Double ML 和 Causal Forest 的 MSE
  
注意: τ_true 无法直接观测，可用 "augmented IPW" 估计
```

### 假设 H3: A+B 生成的序列成功率更高

```
实验设计:
  组 A (基线): 用当前 design_strategy 生成 100 条序列
  组 B (A+B):  用融合嵌入 + Causal Forest 生成 100 条序列
  
  送两组序列做 RF2 结构预测
  
零假设 H0: pass_rate(B) ≤ pass_rate(A)
备择假设 H1: pass_rate(B) > pass_rate(A)

检验方法: 两组比例的 Z 检验
样本量: 每组 100 条
预期功效: 若 pass_rate(A)=12%, pass_rate(B)=25%, α=0.05, 功效 > 80%
```

---

## 六、与后续试验的衔接

### 6.1 产出物清单

| 产出 | 用途 |
|------|------|
| `embedding_t33_650M.npy` | 下游所有预测任务的输入特征 |
| `causal_forest_model.pkl` | 个性化 CATE 估计器 |
| `subgroup_rules.json` | 亚群特异性设计规则 |
| `personalized_suggestions.csv` | 每条失败序列的 Top-3 个性化建议 |

### 6.2 如何用于"生成新蛋白质序列"

**Step 1**: 用融合嵌入训练一个**序列到成功概率**的映射
```
p(success | seq) = sigmoid(MLP(Z_fused(seq)))
```

**Step 2**: 用 Causal Forest 构建**突变效果数据库**
```
对每个 (序列上下文, 突变) 对:
  ΔPAE = CausalForest.predict(Z_fused, mutation)
  → 建立查询表: "在这种上下文中，做这种突变的效果"
```

**Step 3**: 序列优化算法
```
输入: 初始序列 seq_0 (可以是随机或现有失败序列)
重复:
  1. Z = embed_fusion(seq_current)
  2. 评估所有单点突变的 ΔPAE (查 Causal Forest)
  3. 选择 ΔPAE 最小 (最负) 且显著的突变
  4. seq_current = apply_mutation(seq_current, best_mutation)
直到: p(success | seq_current) > 0.8
输出: 优化后的序列
```

**这就是从"分析已有数据"到"主动设计新序列"的关键跃迁。**

---

## 七、技术实现要点

### 7.1 ESM-2 多模型编码实现

```python
import esm
import torch

# 加载三个模型
model_12, alphabet_12 = esm.pretrained.esm2_t12_35M_UR50D()
model_30, alphabet_30 = esm.pretrained.esm2_t30_150M_UR50D()
model_33, alphabet_33 = esm.pretrained.esm2_t33_650M_UR50D()

# t33 需要模型并行 (分到 GPU 2,3)
model_33 = torch.nn.DataParallel(model_33, device_ids=[2, 3])
model_33 = model_33.to('cuda:2')

def extract_fusion_embedding(sequence):
    z12 = model_12.encode(sequence)   # GPU 0,  ~1s
    z30 = model_30.encode(sequence)   # GPU 1,  ~3s
    z33 = model_33.encode(sequence)   # GPU 2-3, ~8s
    return np.concatenate([z12, z30, z33])  # ℝ^2400
```

### 7.2 Causal Forest 实现

```python
from econml.grf import CausalForest

# 准备数据
X = embeddings_fused          # ℝ^(n×2400)
T = treatment_binary          # e.g., first_is_aromatic
Y = outcome_continuous        # e.g., rf2_interaction_pae

# 训练因果森林
est = CausalForest(
    n_estimators=500,       # 500 棵树
    criterion='mse',        # MSE 分裂准则
    max_depth=None,         # 不限制深度
    min_samples_leaf=40,    # 叶节点最少样本 (与数据量匹配)
    n_jobs=8,               # 用满 8 核 CPU
    random_state=42
)
est.fit(X, T, Y)

# 个性化 CATE 预测
cate_individual = est.effect(X_test)
cate_stderr = est.effect_stderr(X_test)  # 标准误

# 显著性筛选
significant_mask = np.abs(cate_individual) > 1.96 * cate_stderr
```

### 7.3 计算成本估算

| 任务 | 当前 (t12+DoubleML) | A+B (融合+CausalForest) | 增加时间 |
|------|-------------------|------------------------|---------|
| ESM-2 编码 | 49s | ~10-12min | +10min |
| Causal Forest | 0s (Double ML ~5s) | ~8-10min | +10min |
| 其他阶段 | ~5min | ~5min | 不变 |
| **总计** | **~6min** | **~25-30min** | **+20min** |

**结论**: 增加 20 分钟计算时间，换取个体化设计能力和显著更高的预测精度。

---

## 八、总结

| 维度 | 基线 (t12 + Double ML) | 方案 A+B (融合 + Causal Forest) |
|------|----------------------|--------------------------------|
| **表示能力** | 480 维局部特征 | 2400 维多尺度特征 |
| **因果推断** | 常数 CATE (一刀切) | 异质性 CATE (个性化) |
| **生成策略** | 硬约束过滤 | 个性化优化 + 置信度评估 |
| **数学工具** | 线性回归 + IPW | 随机森林 + Honest Tree |
| **论文价值** | 描述性分析 | **个体化因果推断 + 生成设计** |
| **计算成本** | ~6 min | ~25-30 min |

**核心改进**: 从"所有人吃同一种药"升级到"精准医疗式个体化设计"。
