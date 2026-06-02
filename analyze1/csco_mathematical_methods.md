# CSC-O 项目数学方法全面总结

---

## 一、项目概述

CSC-O（Causal-Stratified Counterfactual Optimization）是一条面向抗体VH序列设计的因果推理管线，核心目标是从10,572条抗体VH序列中，将最终候选率（final_candidate率）从基线的0.61%提升至5–6%，实现约10倍增幅。

管线由6个阶段组成：

| 阶段 | 名称 | 核心任务 |
|------|------|----------|
| 1 | 数据工程 | 多源数据清洗、特征提取、序列标注 |
| 2 | 分层归因 | 生存分析 + Kaplan-Meier 估计，识别关键风险因子 |
| 3 | 因果约束 | PC算法因果发现 + 后门准则ATE估计，建立因果图 |
| 4 | ESM-2编码 | 蛋白质语言模型提取序列嵌入表示 |
| 5 | 反事实导航 | DML/R-learner/CausalForest估计CATE + 位置特异性Ridge + 单点突变反事实 |
| 6 | 规则合成 | 软偏好阈值 + 反模式检测 + 约束序列生成 |

每个阶段的方法论和数学原理将在下文逐一展开。

---

## 二、生存分析

### 2.1 Cox比例风险模型

#### 数学原理

Cox比例风险模型（Cox Proportional Hazards Model）是半参数生存分析模型，不假设基线风险函数的具体形式，仅对协变量的乘性效应建模。

**模型公式：**

$$h(t \mid X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p)$$

其中：
- $h(t \mid X)$：给定协变量 $X$ 时，在时间 $t$ 的瞬时风险率（hazard rate）
- $h_0(t)$：基线风险函数（baseline hazard），对所有个体相同但形式不指定
- $\beta_j$：第 $j$ 个协变量的回归系数
- $X_j$：第 $j$ 个协变量的取值

**对数风险比：**

$$\ln \frac{h(t \mid X_1)}{h(t \mid X_2)} = \beta^T (X_1 - X_2)$$

这表明风险比仅依赖于协变量差异，与时间无关——即"比例风险"假设。

#### 偏似然估计

由于 $h_0(t)$ 未知，Cox采用偏似然（Partial Likelihood）绕过基线风险：

$$L(\beta) = \prod_{i: \delta_i = 1} \frac{\exp(\beta^T X_i)}{\sum_{j \in R(t_i)} \exp(\beta^T X_j)}$$

其中：
- $\delta_i = 1$ 表示第 $i$ 个个体发生了事件
- $R(t_i)$ 是在时间 $t_i$ 仍处于风险集中的所有个体
- 分子是发生事件个体的风险贡献，分母是风险集中所有个体的风险总和

对数偏似然：

$$\ell(\beta) = \sum_{i: \delta_i = 1} \left[ \beta^T X_i - \ln \sum_{j \in R(t_i)} \exp(\beta^T X_j) \right]$$

通过Newton-Raphson迭代求解 $\hat{\beta}$。

#### 风险比与置信区间

**风险比（Hazard Ratio）：**

$$HR_j = \exp(\beta_j)$$

- $HR > 1$：该协变量增加"死亡"风险（不利因子）
- $HR < 1$：该协变量降低"死亡"风险（保护因子）

**95%置信区间：**

$$CI_{95\%} = \exp\left(\hat{\beta}_j \pm 1.96 \cdot SE(\hat{\beta}_j)\right)$$

#### Concordance Index

C-index衡量模型区分度的能力：

$$C\text{-index} = P(\hat{h}_i > \hat{h}_j \mid T_i < T_j)$$

即：在所有可比事件对中，模型正确预测风险排序的比例。$C = 0.5$ 表示随机预测，$C = 1.0$ 表示完美区分。

#### 应用场景

在CSC-O管线中，抗体设计过程被视为"生存"过程：

| 阶段 | "死亡"事件 | 时间编码 |
|------|-----------|----------|
| RF2结构预测失败 | 阶段1死亡 | time=1 |
| AF3验证失败 | 阶段2死亡 | time=2 |
| PAE筛选未通过 | 阶段3死亡 | time=3 |
| 通过所有阶段 | 删失（censored） | time=3, event=0 |

6个协变量（均值中心化）：
- `glycine_ratio`、`aromatic_ratio`、`proline_ratio`、`cdr3_length`、`mean_esm2_plddt`、`instability_index`

#### 实现细节

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(
    df,
    duration_col='time',
    event_col='event',
    show_progress=True
)
```

#### 关键结果

| 协变量 | $\hat{\beta}$ | HR | 95% CI | p值 |
|--------|--------------|-----|--------|-----|
| glycine_ratio | 1.47 | **4.35** | [2.81, 6.73] | <0.001 |
| aromatic_ratio | -0.40 | **0.67** | [0.52, 0.87] | 0.003 |
| proline_ratio | 0.89 | 2.43 | [1.56, 3.79] | <0.001 |
| cdr3_length | 0.12 | 1.13 | [1.07, 1.19] | <0.001 |

**解读**：甘氨酸比例每增加一个单位，"死亡"风险增加3.35倍（HR=4.35），是最强风险因子；芳香族氨基酸比例则是保护因子（HR=0.67），每增加一个单位风险降低33%。

---

### 2.2 Kaplan-Meier生存估计

#### 数学原理

Kaplan-Meier估计是非参数方法，直接估计生存函数 $S(t) = P(T > t)$。

**公式：**

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

其中：
- $t_i$：第 $i$ 个事件发生时间
- $d_i$：在 $t_i$ 时刻发生事件的个体数
- $n_i$：在 $t_i$ 时刻之前仍处于风险集中的个体数

**Greenwood置信区间：**

$$\text{Var}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{t_i \leq t} \frac{d_i}{n_i(n_i - d_i)}$$

$$CI_{95\%} = \hat{S}(t) \pm 1.96 \cdot \sqrt{\text{Var}[\hat{S}(t)]}$$

#### 应用场景

按CDR3长度分组比较生存曲线：

| 分组 | CDR3长度 | 样本数 | 3阶段通过率 |
|------|---------|--------|------------|
| 短 | 5–7 | ~2,100 | ~8.2% |
| 中 | 8–9 | ~4,200 | ~3.1% |
| 长 | 10+ | ~4,272 | <1% |

对数秩检验（Log-rank test）用于检验组间差异：

$$\chi^2 = \frac{\left(\sum_i (O_i - E_i)\right)^2}{\sum_i \text{Var}(O_i)}$$

#### 实现细节

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
for group in ['5-7', '8-9', '10+']:
    mask = df['cdr3_length_group'] == group
    kmf.fit(df.loc[mask, 'time'],
            event_observed=df.loc[mask, 'event'],
            label=f'CDR3_{group}')
    kmf.plot_survival_function()
```

---

## 三、因果发现

### 3.1 PC算法

#### 数学原理

PC算法（Peter-Clark算法）是基于约束的因果发现方法，从完全连通图出发，通过条件独立性检验逐步删除边，再根据V结构确定方向。

**算法步骤：**

1. **初始化**：构建完全无向图 $G$（所有节点两两相连）
2. **逐层删边**（条件独立性检验）：
   - 深度 $d = 0$：检验 $X_i \perp\!\!\!\perp X_j \mid \emptyset$（边际独立性）
   - 深度 $d = 1$：检验 $X_i \perp\!\!\!\perp X_j \mid \{Z\}$，其中 $|Z| = 1$
   - 深度 $d = 2$：检验 $X_i \perp\!\!\!\perp X_j \mid \{Z_1, Z_2\}$，其中 $|Z| = 2$
   - ...直到条件集大小达到上限
3. **方向确定**（V结构识别）
4. **方向传播**（Meek规则）

#### 偏相关检验

**方法一：Fisher Z变换**

给定偏相关系数 $\rho_{ij \mid Z}$：

$$Z_{ij \mid Z} = \frac{1}{2} \ln \frac{1 + \rho_{ij \mid Z}}{1 - \rho_{ij \mid Z}} \cdot \sqrt{n - |Z| - 3}$$

在零假设 $X_i \perp\!\!\!\perp X_j \mid Z$ 下，$Z_{ij \mid Z} \sim \mathcal{N}(0, 1)$。

若 $|Z_{ij \mid Z}| > \Phi^{-1}(1 - \alpha/2)$，则拒绝独立性假设（保留边）。

**方法二：线性回归残差法**

1. 对条件集 $Z$ 分别回归 $X_i$ 和 $X_j$：

$$\text{res}_x = X_i - \text{LR}(Z \to X_i), \quad \text{res}_y = X_j - \text{LR}(Z \to X_j)$$

2. 计算残差的Pearson相关系数：

$$r = \text{pearson}(\text{res}_x, \text{res}_y)$$

3. 检验统计量：

$$Z = \frac{1}{2} \ln \frac{1 + r}{1 - r} \cdot \sqrt{n - |Z| - 3}$$

#### V结构识别

若 $i - m - j$ 且 $i$ 与 $j$ 不相邻，且 $m$ 不在 $i$ 与 $j$ 的分离集中（即 $m \notin S_{ij}$），则形成collider（V结构）：

$$i \to m \leftarrow j$$

#### 应用场景

在CSC-O中，PC算法用于发现序列特征之间的因果结构：

- **节点**：glycine_ratio, aromatic_ratio, proline_ratio, cdr3_length, mean_esm2_plddt, instability_index, RF2_pass, AF3_pass, PAE_pass
- **条件集深度**：0到3
- **显著性水平**：$\alpha = 0.01$
- **领域约束**：强制treatment → outcome方向（如 glycine_ratio → PAE_pass）

#### 实现细节

```python
from causallearn.search.ConstraintBased.PC import pc

cg = pc(data, alpha=0.01, indep_test='fisherz', max_cond_set=3)
```

领域约束通过后处理强制方向实现：若发现边 `PAE_pass → glycine_ratio`，则翻转方向为 `glycine_ratio → PAE_pass`。

---

### 3.2 后门准则与ATE估计

后门准则（Back-door Criterion）确保我们可以通过调整混杂变量来识别因果效应。若变量集 $Z$ 满足：
1. $Z$ 阻断从 $T$ 到 $Y$ 的所有后门路径
2. $Z$ 不包含 $T$ 的后代

则平均处理效应（ATE）可识别。

$$ATE = E[Y(1) - Y(0)] = E_X\left[E[Y \mid T=1, X] - E[Y \mid T=0, X]\right]$$

#### 3.2.1 逆概率加权（IPW）

##### 数学原理

**倾向得分（Propensity Score）：**

$$e(X) = P(T = 1 \mid X)$$

即给定协变量 $X$ 下接受处理的概率。

**IPW权重：**

$$W_i = \frac{T_i}{e(X_i)} + \frac{1 - T_i}{1 - e(X_i)}$$

**ATE估计：**

$$\hat{ATE}_{IPW} = \frac{1}{n} \sum_{i=1}^{n} \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1 - T_i) Y_i}{1 - e(X_i)} \right]$$

等价于：

$$\hat{ATE}_{IPW} = E_n\left[\frac{Y \cdot T}{e(X)}\right] - E_n\left[\frac{Y \cdot (1-T)}{1 - e(X)}\right]$$

**直觉**：通过对少数群体（如处理组中倾向得分低的个体）赋予更大权重，平衡处理组和对照组的协变量分布，模拟随机化实验。

##### 实现细节

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ps_model = LogisticRegression(max_iter=1000)
ps = cross_val_predict(ps_model, X_conf, T, cv=5, method='predict_proba')[:, 1]
ps = np.clip(ps, 0.01, 0.99)

ipw = np.where(T == 1, T / ps, (1 - T) / (1 - ps))
ipw = np.clip(ipw, 0, 10)

ate_ipw = np.average(Y[T == 1], weights=ipw[T == 1]) - \
          np.average(Y[T == 0], weights=ipw[T == 0])
```

**关键参数**：
- 倾向得分截断：`clip(0.01, 0.99)` —— 防止极端权重
- IPW权重截断：`clip(0, 10)` —— 限制最大权重，降低方差
- 5折交叉验证预测倾向得分 —— 避免过拟合

---

#### 3.2.2 线性回归调整

##### 数学原理

**模型：**

$$Y = \alpha + \beta \cdot T + \gamma^T X_{\text{conf}} + \varepsilon$$

其中：
- $T$：处理变量（如 glycine_ratio 是否高于中位数）
- $X_{\text{conf}}$：混杂变量集
- $\beta$：**即ATE的估计**（处理变量系数）

**OLS估计：**

$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

**标准误：**

$$SE(\hat{\beta}) = \sqrt{MSE \cdot \left[(X^T X)^{-1}\right]_{[0,0]}}$$

其中 $MSE = \frac{1}{n - p} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$。

**t检验：**

$$t = \frac{\hat{\beta}}{SE(\hat{\beta})}, \quad p = 2\left(1 - \Phi(|t|)\right)$$

**95%置信区间：**

$$CI_{95\%} = \hat{\beta} \pm 1.96 \cdot SE(\hat{\beta})$$

##### 实现细节

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_design, Y)

beta = lr.coef_[0]
XtX_inv = np.linalg.pinv(X_design.T @ X_design)
mse = np.sum((Y - lr.predict(X_design)) ** 2) / (n - p)
se = np.sqrt(mse * XtX_inv[0, 0])
t_stat = beta / se
p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
ci = (beta - 1.96 * se, beta + 1.96 * se)
```

##### 关键结果

以 glycine_ratio（二值化）对 PAE 的影响为例：

| 方法 | ATE | 95% CI | p值 |
|------|-----|--------|-----|
| IPW | -4.52 | [-6.81, -2.23] | <0.001 |
| 线性回归 | **-4.69** | [-6.94, -2.44] | <0.001 |

ATE为负值意味着高甘氨酸组PAE更低（结构更差），与Cox模型HR=4.35一致。

---

#### 3.2.3 分层ATE

##### 数学原理

按分层变量 $S$（如CDR3长度）将样本分为 $K$ 层，在每层内独立估计ATE：

$$\hat{ATE}_k = \frac{1}{n_k} \sum_{i \in \text{stratum}_k} \left[\frac{T_i Y_i}{e_k(X_i)} - \frac{(1-T_i) Y_i}{1 - e_k(X_i)}\right]$$

总体ATE：

$$\hat{ATE}_{\text{stratified}} = \sum_{k=1}^{K} \frac{n_k}{n} \cdot \hat{ATE}_k$$

##### 应用场景

按CDR3长度分层，识别因果效应的异质性：

| CDR3长度层 | 样本数 | ATE(glycine→PAE) | p值 |
|-----------|--------|-------------------|-----|
| 5–7 | ~2,100 | -2.31 | 0.042 |
| 8–9 | ~4,200 | -4.89 | <0.001 |
| 10+ | ~4,272 | -6.73 | <0.001 |

**解读**：CDR3越长，甘氨酸对PAE的负面影响越强，说明因果效应存在异质性。

##### 实现细节

- 最小样本量约束：全层 $\geq 50$，子层（处理/对照）$\geq 30$
- 不满足约束的层合并或排除
- 每层内使用IPW或线性回归估计ATE

---

## 四、异质性因果效应估计

### 4.1 Double Machine Learning (DML)

DML的核心思想：用机器学习模型灵活地估计混淆函数，再通过残差回归获得 $\sqrt{n}$-一致的因果效应估计。

#### 4.1.1 部分线性回归（PLR）

##### 数学原理

**模型设定：**

$$Y = \theta \cdot T + g(X) + \varepsilon, \quad \varepsilon \perp\!\!\!\perp (X, T)$$

$$T = m(X) + \eta, \quad \eta \perp\!\!\!\perp X$$

其中：
- $\theta$：目标因果参数（ATE）
- $g(X)$：结果模型的混淆函数（非线性）
- $m(X)$：处理模型的混淆函数（倾向函数的非参数推广）

**交叉拟合步骤（Cross-fitting）：**

将样本分为 $K$ 折（如5折），对每折 $k$：

1. 在 $K \setminus k$ 上训练 $\hat{g}^{(-k)}(X)$ 和 $\hat{m}^{(-k)}(X)$
2. 在第 $k$ 折上计算残差：

$$\tilde{Y}_i = Y_i - \hat{g}^{(-k)}(X_i), \quad \tilde{T}_i = T_i - \hat{m}^{(-k)}(X_i)$$

**第二步：残差回归**

$$\hat{\theta} = \frac{\sum_{i=1}^{n} \tilde{T}_i \cdot \tilde{Y}_i}{\sum_{i=1}^{n} \tilde{T}_i^2}$$

**标准误：**

$$SE(\hat{\theta}) = \frac{\sqrt{\frac{1}{n-1} \sum_{i=1}^{n} \hat{\varepsilon}_i^2}}{\sqrt{\sum_{i=1}^{n} \tilde{T}_i^2}}$$

其中 $\hat{\varepsilon}_i = \tilde{Y}_i - \hat{\theta} \cdot \tilde{T}_i$。

**为什么有效**：交叉拟合消除了正则化偏差——$\hat{g}$ 和 $\hat{m}$ 的训练数据与预测数据独立，因此残差中的过拟合偏差为零。

##### 实现细节

```python
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, LGBMClassifier

kf = KFold(n_splits=5, shuffle=True, random_state=42)
Y_resid = np.zeros(n)
T_resid = np.zeros(n)

for train_idx, test_idx in kf.split(X):
    model_y = LGBMRegressor(n_estimators=100, max_depth=5)
    model_t = LGBMClassifier(n_estimators=100, max_depth=5)
    model_y.fit(X[train_idx], Y[train_idx])
    model_t.fit(X[train_idx], T[train_idx])
    Y_resid[test_idx] = Y[test_idx] - model_y.predict(X[test_idx])
    T_resid[test_idx] = T[test_idx] - model_t.predict_proba(X[test_idx])[:, 1]

theta_hat = np.sum(T_resid * Y_resid) / np.sum(T_resid ** 2)
```

---

#### 4.1.2 R-learner

##### 数学原理

R-learner基于Robinson变换，直接优化异质性因果效应。

**Robinson变换：**

由 $Y = \theta(X) \cdot T + g(X) + \varepsilon$ 和 $T = m(X) + \eta$，可得：

$$Y - g(X) = \theta(X) \cdot (T - m(X)) + \varepsilon$$

令 $\tilde{Y} = Y - g(X)$，$\tilde{T} = T - m(X)$：

$$\tilde{Y} = \theta(X) \cdot \tilde{T} + \varepsilon$$

**R-learner目标函数：**

$$\hat{\theta}(\cdot) = \arg\min_{\theta} \left\{ \frac{1}{n} \sum_{i=1}^{n} \left( \tilde{Y}_i - \theta(X_i) \cdot \tilde{T}_i \right)^2 \right\}$$

**逐样本CATE估计（瞬时解）：**

$$\tilde{\tau}_i = \frac{\tilde{Y}_i \cdot \tilde{T}_i}{\tilde{T}_i^2 + \epsilon}$$

其中 $\epsilon$ 是数值稳定项（如 $10^{-6}$）。

**异质性建模：**

用梯度提升回归（GBR）拟合逐样本CATE：

$$\text{GBR.fit}(X, \tilde{\tau}) \implies \widehat{CATE}(x) = \text{GBR.predict}(x)$$

##### 实现细节

```python
from sklearn.ensemble import GradientBoostingRegressor

ps = cross_val_predict(LGBMClassifier(), X, T, cv=5, method='predict_proba')[:, 1]
ps = np.clip(ps, 0.1, 0.9)

m_x = cross_val_predict(LGBMRegressor(), X, T, cv=5)
g_x = cross_val_predict(LGBMRegressor(), X, Y, cv=5)

Y_tilde = Y - g_x
T_tilde = T - m_x

tau_tilde = Y_tilde * T_tilde / (T_tilde ** 2 + 1e-6)
tau_tilde = np.clip(tau_tilde, -50, 50)

gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
gbr.fit(X, tau_tilde)
cate = gbr.predict(X)
```

**关键参数**：
- 倾向得分截断：`clip(ps, 0.1, 0.9)` —— 防止极端权重
- CATE截断：`clip(cate, -50, 50)` —— 防止异常值
- GBR参数：`n_estimators=200, max_depth=4, learning_rate=0.1`

---

#### 4.1.3 Causal Forest (CausalForestDML)

##### 数学原理

因果森林（Causal Forest）是基于随机森林的非参数CATE估计方法。

**核心思想**：在每棵树的每个叶节点内，通过局部矩估计计算ATE，然后对所有树取平均得到CATE。

**分裂准则**：不同于普通随机森林最小化MSE，因果森林最大化**因果效应异质性**：

$$\max_{\text{split}} \widehat{\text{Var}}(\hat{\tau}_L, \hat{\tau}_R)$$

其中 $\hat{\tau}_L$、$\hat{\tau}_R$ 分别是左右子节点的局部ATE估计。

**局部ATE估计（叶节点内）：**

$$\hat{\tau}_{\text{leaf}} = \frac{\sum_{i \in \text{leaf}} \tilde{Y}_i \cdot \tilde{T}_i}{\sum_{i \in \text{leaf}} \tilde{T}_i^2}$$

**逐样本CATE：**

$$\widehat{CATE}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{\tau}_{L_b(x)}$$

其中 $L_b(x)$ 是第 $b$ 棵树中 $x$ 所在的叶节点。

**推断标准误**：基于局部矩估计的渐近正态性，提供逐样本置信区间。

##### 实现细节

```python
from econml.dml import CausalForestDML

cf = CausalForestDML(
    model_y=LGBMRegressor(n_estimators=100, max_depth=5),
    model_t=LGBMClassifier(n_estimators=100, max_depth=5),
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=20,
    cv=3,
    random_state=42
)
cf.fit(Y, T, X=X, W=None)
cate = cf.effect(X)
cate_se = cf.effect_inference(X).stderr
```

**降级策略**：

```
CausalForestDML (econml可用)
    ↓ 降级
R-learner (LGBM + GBR)
    ↓ 降级
PLR (LGBM + OLS残差回归)
```

##### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 200 | 树的数量 |
| max_depth | 4 | 最大深度，防止过拟合 |
| min_samples_leaf | 20 | 叶节点最小样本数 |
| cv | 3 | 交叉拟合折数 |

---

### 4.2 位置特异性CATE（Ridge回归）

##### 数学原理

**模型设定：**

$$PAE_i = \alpha + \beta \cdot \mathbb{I}(\text{aa@pos}) + \gamma^T \cdot \text{ESM2\_embed}_i + \varepsilon_i$$

其中：
- $\mathbb{I}(\text{aa@pos})$：氨基酸类型在特定位置的指示变量（如 "W@100" = 1表示位置100是色氨酸）
- $\beta$：**即该位置该氨基酸的CATE**
- $\text{ESM2\_embed}_i$：ESM-2序列嵌入（控制序列全局信息）

**Ridge正则化：**

$$\hat{\beta}_{\text{Ridge}} = \arg\min_{\beta} \left\{ \|Y - X\beta\|^2 + \lambda \|\beta\|^2 \right\}$$

解析解：

$$\hat{\beta}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T Y$$

**标准误**（忽略Ridge偏差的近似）：

$$SE(\hat{\beta}_j) = \sqrt{MSE \cdot \left[(X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1}\right]_{[j,j]}}$$

简化近似（与OLS相同公式）：

$$SE(\hat{\beta}_j) \approx \sqrt{MSE \cdot \left[(X^T X)^{-1}\right]_{[j,j]}}$$

**显著性筛选**：$|t| > 2.0$（近似对应 $p < 0.05$）

##### 应用场景

为每个CDR3位置 × 每种氨基酸类型估计独立的因果效应，生成"位置特异性CATE查找表"：

| 位置 | 氨基酸 | CATE ($\beta$) | SE | t值 | 显著 |
|------|--------|----------------|-----|-----|------|
| 100 | W | -3.21 | 0.89 | -3.61 | ✓ |
| 100 | G | +5.47 | 1.12 | +4.88 | ✓ |
| 101 | Y | -2.83 | 0.94 | -3.01 | ✓ |
| 101 | P | +4.12 | 1.35 | +3.05 | ✓ |

**解读**：位置100处色氨酸(W)降低PAE 3.21（改善结构），甘氨酸(G)增加PAE 5.47（恶化结构）。

##### 实现细节

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_design, Y)

beta = ridge.coef_[aa_indicator_indices]
XtX = X_design.T @ X_design
mse = np.sum((Y - ridge.predict(X_design)) ** 2) / (n - p)
XtX_inv = np.linalg.pinv(XtX)
se = np.sqrt(mse * np.diag(XtX_inv)[aa_indicator_indices])
t_stat = beta / se
```

**最小样本约束**：
- 位置覆盖 $\geq 100$ 条序列
- 特定氨基酸在该位置出现 $\geq 30$ 次

---

## 五、蛋白质语言模型

### 5.1 ESM-2 Transformer编码

##### 数学原理

ESM-2（Evolutionary Scale Modeling 2）是基于Transformer encoder架构的蛋白质语言模型，在海量蛋白质序列上自监督预训练。

**Transformer Encoder核心计算：**

**多头自注意力（Multi-Head Self-Attention）：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

$$\text{MultiHead}(H) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(H W_i^Q, H W_i^K, H W_i^V)$$

**前馈网络（FFN）：**

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

**层归一化 + 残差连接：**

$$H^{(l+1)} = \text{LayerNorm}(H^{(l)} + \text{MultiHead}(H^{(l)}))$$

$$H^{(l+1)} = \text{LayerNorm}(H^{(l+1)} + \text{FFN}(H^{(l+1)}))$$

**模型规模：**

| 模型 | 层数 | 隐藏维度 | 注意力头数 | 参数量 |
|------|------|---------|-----------|--------|
| esm2_t12_35M | 12 | 480 | 12 | 35M |
| esm2_t30_150M | 30 | 640 | 20 | 150M |
| esm2_t33_650M | 33 | 1280 | 20 | 650M |
| esm2_t36_3B | 36 | 2560 | 40 | 3B |

**输入处理：**

氨基酸序列 → token化（含特殊token `[CLS]` 和 `[EOS]`）：

$$\text{tokens} = [\text{[CLS]}, \text{aa}_1, \text{aa}_2, \ldots, \text{aa}_L, \text{[EOS]}]$$

**输出提取：**

- Token级表示：$\{h_0, h_1, \ldots, h_L, h_{L+1}\}$（$h_0$ = `[CLS]`）
- 序列级嵌入 = mean pooling：

$$\text{embed}_{\text{seq}} = \frac{1}{L} \sum_{i=1}^{L} h_i$$

即对 `[CLS]` 和 `[EOS]` 之外的token表示取均值。

**多模型融合策略：**

| 策略 | 公式 | 维度 |
|------|------|------|
| 拼接 | $\text{embed} = [\text{embed}_{35M}; \text{embed}_{150M}]$ | 480 + 640 = 1120 |
| 平均 | $\text{embed} = \frac{1}{2}(\text{embed}_{35M} + \text{embed}_{150M})$ | 480（需对齐维度） |
| PCA+拼接 | $\text{embed} = [\text{PCA}(\text{embed}_{35M}); \text{PCA}(\text{embed}_{150M})]$ | 可控 |

##### 实现细节

```python
import esm

model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

with torch.no_grad():
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    results = model(batch_tokens, repr_layers=[12])
    token_repr = results["representations"][12]
    seq_embed = token_repr[:, 1:seq_len+1, :].mean(dim=1)
```

---

### 5.2 降维可视化

#### t-SNE

**SNE（Stochastic Neighbor Embedding）：**

高维空间中的相似度：

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

低维空间中的相似度：

$$q_{j|i} = \frac{\exp(-\|y_i - y_j\|^2)}{\sum_{k \neq i} \exp(-\|y_i - y_k\|^2)}$$

**对称化：**

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**目标函数（KL散度）：**

$$C = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**参数**：perplexity=30（控制有效邻居数），采样3000条序列。

#### PCA

**协方差矩阵：**

$$C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}$$

**特征分解：**

$$C = V \Lambda V^T$$

**投影：**

$$Z = X_{\text{centered}} \cdot V_{[:, :k]}$$

其中 $k = \min(256, \text{dim}, n)$。

---

## 六、聚类与亚群发现

### 6.1 KMeans

##### 数学原理

**目标函数：**

$$\min_{\{c_k\}} \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2$$

其中 $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$ 是第 $k$ 个簇的质心。

**迭代步骤（Lloyd算法）：**

1. **分配**：$C_k = \{i : k = \arg\min_j \|x_i - \mu_j\|^2\}$
2. **更新**：$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$
3. 重复直到收敛

##### 应用场景

输入：CATE值标准化后的1维特征（即 $\hat{\tau}_i / \text{std}(\hat{\tau})$）

参数：`n_clusters=3, n_init=10`

**亚群画像：**

| 亚群 | size | pass_rate | FC_rate | mean_PAE | mean_CATE | 特征均值 |
|------|------|-----------|---------|----------|-----------|---------|
| 高风险 | ~3,500 | 0.3% | 0.1% | 18.7 | +5.2 | 高gly, 低aro |
| 中风险 | ~4,000 | 1.2% | 0.5% | 13.4 | +1.1 | 中gly, 中aro |
| 低风险 | ~3,072 | 3.8% | 1.8% | 9.2 | -2.3 | 低gly, 高aro |

---

### 6.2 最近邻检索

##### 数学原理

**余弦距离：**

$$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \cdot \|y\|} = 1 - \cos\theta$$

其中 $\theta$ 是向量 $x$ 和 $y$ 的夹角。

余弦距离范围 $[0, 2]$：
- $d = 0$：方向完全相同
- $d = 1$：正交
- $d = 2$：方向完全相反

##### 应用场景

为失败序列（PAE > 10）在成功序列集中寻找最近邻模板：

- `n_neighbors=5`
- 距离度量：余弦距离（基于ESM-2嵌入）
- 输出：5个最相似的成功序列及其关键特征差异

---

## 七、反事实推理与建议生成

### 7.1 单点突变反事实

##### 数学原理

反事实推理的核心问题：**如果序列在位置 $p$ 的氨基酸从 $a$ 变为 $a'$，PAE会如何变化？**

**枚举策略：**

对每条序列的CDR3区域（约13个位置），枚举所有20种氨基酸替换：

$$\text{总突变数} \approx 13 \times 20 = 260 \text{种/序列}$$

**PAE预测（查表法）：**

$$\Delta PAE(p, a \to a') = \widehat{CATE}(a'@p) - \widehat{CATE}(a@p)$$

其中 $\widehat{CATE}(a@p)$ 来自位置特异性Ridge回归的查找表。

**排序准则：**

按 $\Delta PAE$ 升序排列（负值 = PAE降低 = 结构改善）：

$$\text{Top-}K = \text{argsort}(\Delta PAE)[:K], \quad K = 3$$

**编辑距离（Hamming距离）：**

$$d_H(s_1, s_2) = \sum_{i=1}^{L} \mathbb{I}(s_{1,i} \neq s_{2,i})$$

单点突变的Hamming距离恒为1。

##### 实现细节

```python
for seq_id, row in df.iterrows():
    cdr3 = row['cdr3']
    suggestions = []
    for pos in range(len(cdr3)):
        current_aa = cdr3[pos]
        for new_aa in AMINO_ACIDS:
            if new_aa == current_aa:
                continue
            delta_pae = position_specific_cate.get((pos, new_aa), 0) - \
                        position_specific_cate.get((pos, current_aa), 0)
            suggestions.append((pos, current_aa, new_aa, delta_pae))
    suggestions.sort(key=lambda x: x[3])
    top3 = suggestions[:3]
```

---

### 7.2 截短建议

##### 策略

对CDR3长度 $\geq 10$ 的序列，建议截短至6或7个残基：

$$\text{CDR3}_{\text{truncated}} = \text{CDR3}[:6] \quad \text{或} \quad \text{CDR3}[:7]$$

##### 依据

| CDR3长度 | 通过率 | 样本数 |
|---------|--------|--------|
| 5–7 | **51.6%** | ~2,100 |
| 8–9 | 12.3% | ~4,200 |
| 10+ | <3% | ~4,272 |

截短建议覆盖883条长CDR3序列。

##### 风险

截短改变了CDR3的生物学功能，需结合结构验证确认截短后仍保持抗原结合能力。

---

## 八、规则合成

### 8.1 软偏好阈值搜索

##### 数学原理

**贪心策略**：从严格阈值到宽松阈值逐步扫描，选择满足通过率约束的最严格阈值。

**形式化**：

给定特征 $f$ 和阈值候选集 $\Theta_f = \{\theta_1, \theta_2, \ldots, \theta_K\}$（从严格到宽松排列），寻找：

$$\theta^* = \max\{\theta \in \Theta_f : \text{pass\_rate}(f \geq \theta) \geq r_{\min}\}$$

其中 $r_{\min}$ 是最低通过率要求。

**示例**：aromatic_ratio阈值搜索

| 阈值 | 满足条件子集大小 | 通过率 | 是否满足 $r_{\min}=0.15$ |
|------|----------------|--------|--------------------------|
| 0.15 | 6,842 | 0.18 | ✓ |
| 0.20 | 4,217 | 0.22 | ✓ |
| 0.25 | 1,893 | 0.28 | ✓ |
| 0.30 | 412 | 0.35 | ✓（但子集过小） |

选择 $\theta^* = 0.25$：在保持足够子集大小的同时最大化通过率。

##### 实现细节

```python
thresholds = [0.15, 0.20, 0.25]
min_pass_rate = 0.15

for feat, thresh_list in threshold_config.items():
    for thresh in thresh_list:
        subset = df[df[feat] >= thresh]
        pass_rate = subset['final_candidate'].mean()
        if pass_rate >= min_pass_rate and len(subset) >= 100:
            rules.append({'feature': feat, 'threshold': thresh,
                         'pass_rate': pass_rate, 'subset_size': len(subset)})
```

---

### 8.2 反模式检测

##### 数学原理

**准则**：若含模式 $P$ 的子集通过率 < 不含模式子集通过率的50%，则 $P$ 为反模式。

$$\text{AntiPattern}(P) \iff \frac{\text{pass\_rate}(\text{contains } P)}{\text{pass\_rate}(\text{not contains } P)} < 0.5$$

##### 检测结果

| 反模式 | 含模式通过率 | 不含模式通过率 | 比值 |
|--------|------------|--------------|------|
| GGG | 0.08% | 0.72% | 0.11 |
| SSS | 0.12% | 0.68% | 0.18 |
| LL | 0.15% | 0.65% | 0.23 |

**解读**：连续甘氨酸（GGG）是最强反模式，含GGG的序列通过率仅为不含序列的11%。

---

### 8.3 约束序列生成

##### 数学原理

**生成策略**：模板变异（30%概率）+ 约束采样（70%概率）

**模板变异**：

从成功序列模板出发，以概率 $p_{\text{mut}}=0.3$ 在随机位置替换为约束采样氨基酸。

**约束采样**：

基于软偏好权重的氨基酸采样分布：

$$P(aa) \propto \text{base\_freq}(aa) \times \text{preference}(aa)$$

**偏好权重：**

| 氨基酸类别 | 偏好方向 | 权重调整 |
|-----------|---------|---------|
| 芳香族 (F, W, Y) | ↑ | ×1.5 |
| 甘氨酸 (G) | ↓ | ×0.3 |
| 丝氨酸 (S) | ↓ | ×0.5 |
| 脯氨酸 (P) | ↓ | ×0.4 |
| 疏水性 (I, L, V, M) | ↑ | ×1.3 |

**首尾残基白名单**：

CDR3首尾残基限制为特定氨基酸集合（如首位：C, A, S, D；末位：C, W, F, Y），确保结构稳定性。

**多样性过滤**：

1. 余弦相似度过滤：$\cos(\text{embed}_i, \text{embed}_j) < 0.99$
2. Hamming距离过滤：$d_H(s_i, s_j) \geq 3$

确保生成序列之间具有足够多样性。

---

## 九、模型评估

### 9.1 分类评估指标

##### 数学公式

**混淆矩阵元素**：TP, FP, FN, TN

**核心指标：**

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**AUC-ROC**：

$$\text{AUC} = P(\hat{p}_+ > \hat{p}_-)$$

即随机正样本得分高于随机负样本得分的概率。

**Average Precision (AP)**：

$$AP = \sum_{k=1}^{n} P(k) \cdot \Delta R(k)$$

即Precision-Recall曲线下面积，对不平衡数据集比AUC-ROC更有信息量。

##### 评估方案

- 5折StratifiedKFold交叉验证（保持每折中正负样本比例一致）
- 阈值扫描：概率阈值 $\in [0.1, 0.15, 0.2, \ldots, 0.5]$

---

### 9.2 阈值敏感性分析

##### 数学原理

分析RF2 `pred_lddt` 阈值对最终筛选结果的影响。

**扫描参数**：

- RF2 pred_lddt阈值：$\theta \in [0.82, 0.825, 0.830, \ldots, 0.92]$（步长0.005）
- 固定PAE阈值：$\leq 10.0$

**评估指标**：

$$\text{Sensitivity} = \frac{|\text{pass both}|}{|\text{true positive}|}$$

$$\text{Precision} = \frac{|\text{pass both} \cap \text{true positive}|}{|\text{pass both}|}$$

$$\text{Pass Rate} = \frac{|\text{pass both}|}{n_{\text{total}}}$$

**目的**：找到sensitivity和precision的平衡点，确定最优RF2阈值。

---

## 十、方法间逻辑关系

以下是CSC-O管线中各数学方法的逻辑关系与依赖图：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CSC-O 管线方法关系图                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  数据工程     │  原始序列 + 结构预测结果 + 特征提取                  │
│  │  (Stage 1)   │  → 10572条VH序列, 6个协变量, PAE, pLDDT            │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  分层归因     │  ┌─ Cox PH ────→ HR=4.35(gly), HR=0.67(aro)      │
│  │  (Stage 2)   │  │                                             │
│  │              │  └─ Kaplan-Meier ──→ CDR3长度分层生存曲线          │
│  └──────┬───────┘                                                   │
│         │ 识别关键风险因子 → 指导因果变量选择                          │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  因果约束     │  ┌─ PC算法 ────→ 因果图 (DAG)                     │
│  │  (Stage 3)   │  │                                             │
│  │              │  └─ 后门准则 ──┬─ IPW ─────→ ATE=-4.52            │
│  │              │               ├─ 线性回归 ─→ ATE=-4.69            │
│  │              │               └─ 分层ATE ──→ 异质性效应            │
│  └──────┬───────┘                                                   │
│         │ 确认因果方向 + 估计平均效应 → 指导异质性分析                 │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  ESM-2编码    │  ┌─ Transformer ─→ 序列嵌入 (480-2560维)          │
│  │  (Stage 4)   │  │                                             │
│  │              │  ├─ PCA ────────→ 降维 (256维)                    │
│  │              │  └─ t-SNE ──────→ 可视化 (2维)                    │
│  └──────┬───────┘                                                   │
│         │ 嵌入表示 → 作为DML/聚类的特征输入                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  反事实导航   │  ┌─ DML/PLR ────→ ATE (全局)                     │
│  │  (Stage 5)   │  │                                             │
│  │              │  ├─ R-learner ──→ 逐样本CATE                     │
│  │              │  │                                             │
│  │              │  ├─ CausalForest → CATE + 标准误                 │
│  │              │  │                                             │
│  │              │  ├─ Ridge位置CATE → 查找表 (pos×aa)              │
│  │              │  │                                             │
│  │              │  ├─ KMeans ────→ 亚群划分 (3群)                  │
│  │              │  │                                             │
│  │              │  ├─ 最近邻检索 ─→ 成功模板匹配                    │
│  │              │  │                                             │
│  │              │  └─ 反事实生成 ─┬─ 单点突变 (top3/序列)           │
│  │              │                └─ 截短建议 (883条)                │
│  └──────┬───────┘                                                   │
│         │ CATE + 突变建议 → 指导规则设计                              │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  规则合成     │  ┌─ 软偏好阈值 ─→ 特征约束规则                     │
│  │  (Stage 6)   │  │                                             │
│  │              │  ├─ 反模式检测 ─→ 排除规则 (GGG, SSS, LL)         │
│  │              │  │                                             │
│  │              │  └─ 约束生成 ──┬─ 模板变异 (30%)                  │
│  │              │               ├─ 约束采样 (70%)                   │
│  │              │               └─ 多样性过滤 (cos<0.99, dH≥3)      │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  模型评估     │  ┌─ 5折CV分类评估 (AUC, AP, F1)                  │
│  │  (贯穿全程)   │  │                                             │
│  │              │  └─ 阈值敏感性分析 (RF2 lddt扫描)                  │
│  └──────────────┘                                                   │
│                                                                     │
│  ═════════════════════════════════════════════════════════════════   │
│  方法依赖链:                                                         │
│                                                                     │
│  Cox PH ──→ 变量选择 ──→ PC算法 ──→ 因果图 ──→ 后门准则             │
│       │                                    │                        │
│       └──→ 分层变量 ──→ 分层ATE ←─────────┘                        │
│                          │                                          │
│  ESM-2 ──→ 嵌入 ──→ DML/R-learner/CausalForest ──→ CATE            │
│              │                                    │                  │
│              ├─→ PCA ──→ KMeans ──────────────────┘                  │
│              │                      │                                │
│              └─→ 最近邻检索 ←───────┘                                │
│                                                                     │
│  CATE ──→ Ridge位置CATE ──→ 反事实突变建议 ──→ 规则合成             │
│  CATE ──→ KMeans亚群 ──→ 亚群画像 ──→ 规则合成                     │
│                                                                     │
│  目标: 0.61% ──────→ 5-6% (10倍增幅)                                │
└─────────────────────────────────────────────────────────────────────┘
```

**核心逻辑总结**：

1. **生存分析**（Cox + KM）从宏观层面识别关键风险因子，回答"哪些特征重要"
2. **因果发现**（PC算法）建立特征间的因果结构，回答"因果方向是什么"
3. **ATE估计**（IPW + 回归调整 + 分层）量化平均因果效应，回答"效应有多大"
4. **异质性分析**（DML + R-learner + CausalForest）揭示条件因果效应，回答"对谁有效"
5. **位置特异性CATE**（Ridge）精确定位到序列位置和氨基酸类型，回答"改哪里"
6. **反事实推理**（突变枚举 + 截短）生成具体修改建议，回答"怎么改"
7. **规则合成**（软偏好 + 反模式 + 约束生成）将发现转化为可执行的序列设计规则

整个管线遵循"**发现→验证→定位→行动**"的因果推理闭环，每一步的输出都是下一步的输入，确保从统计关联到因果机制的严格推理链条。
