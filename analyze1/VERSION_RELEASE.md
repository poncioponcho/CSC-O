# CSC-O v2.0 正式版本发布说明

## 一、版本信息
- 版本号: v2.0
- 发布日期: 2026-06-02
- 基于数据: Q02223靶点, 10572条抗体VH序列
- 优化目标: final_candidate率从0.61%提升至3.5-6.0%

## 二、版本亮点
- 完整6阶段管线：数据工程→分层归因→因果约束→ESM-2编码→反事实导航→规则合成
- 因果推断框架：Cox PH + PC算法 + ATE/IPW + Double ML (CausalForest/R-learner/PLR)
- 修复了 _double_ml_cate 和 _run_multi_treatment_cate 的卡死问题
- 8个treatment变量全部成功完成CATE估计
- 输出完整设计策略文件（JSON + TXT + 报告）

## 三、脱敏处理范围
详细列出所有脱敏操作：

| 文件 | 脱敏内容类型 | 替换为 |
|------|-------------|--------|
| ENGINEERING_ANALYSIS_REPORT.md | 服务器IP | REDACTED_IP |
| ENGINEERING_ANALYSIS_REPORT.md | 用户名+IP | REDACTED_USER@REDACTED_IP |
| ENGINEERING_ANALYSIS_REPORT.md | 主机名 | REDACTED_HOSTNAME |
| PLAN_AB_IMPLEMENTATION.md | 服务器IP和用户名 | REDACTED_IP / REDACTED_USER@REDACTED_IP |
| PROJECT_PROGRESS_AND_PLAN.md | GitHub用户名 | REDACTED_USER |
| PROJECT_PROGRESS_AND_PLAN.md | 主机名 | REDACTED_HOSTNAME |
| vpn_diagnose.sh | 内网IP | REDACTED_IP |
| tools/ssh_auto_attach.sh | 默认用户名 | REDACTED_USER |
| tools/check_server_status.sh | 默认用户名 | REDACTED_USER |
| tools/vpn_watchdog.sh | 默认用户名 | REDACTED_USER |
| tools/README.md | 默认用户名 | REDACTED_USER |
| .gitignore | 新增排除 output_server*/, *.csv, *.xlsx | - |

说明：含明文密码的 服务器配置记录.md 已被 .gitignore 排除，不会提交到仓库。

## 四、使用指南

### 4.1 环境要求
- Python 3.9+
- 依赖: pandas, numpy, scikit-learn, lifelines, lightgbm, matplotlib, seaborn, esm, torch
- 可选: econml (CausalForestDML, 若不可用自动降级到R-learner)
- GPU: 可选 (ESM-2编码加速)

### 4.2 快速开始
```bash
# 基本运行
python csco_pipeline.py -i your_data.csv -o ./output

# 推荐参数
python csco_pipeline.py -i your_data.csv -o ./output \
    --cate-method r_learner \
    --target final_candidate \
    --resume

# 完整参数
python csco_pipeline.py -i your_data.csv -o ./output \
    --cate-method r_learner \
    --target final_candidate \
    --batch-size 8 \
    --top-n 2000 \
    --esm2-models esm2_t12_35M_UR50D \
    --esm2-fusion concat \
    --subgroup-clusters 3 \
    --resume
```

### 4.3 输入数据格式
- CSV/XLSX/TSV格式
- 必需列: vh_sequence, rf2_pred_lddt, rf2_interaction_pae, rf2_passed_filter
- 可选列: final_candidate, af3_passed_filter, schrodinger_passed_filter 等
- 缺失列会自动补全默认值

### 4.4 输出文件说明
| 文件 | 说明 |
|------|------|
| feature_matrix.csv | 特征矩阵 |
| survival_data.csv | 生存分析数据 |
| cox_hazard_ratios.csv | Cox风险比 |
| ate_estimates.csv | ATE估计 |
| stratified_ate_estimates.csv | 分层ATE |
| multi_treatment_cate.csv | 多treatment CATE |
| position_specific_cate.csv | 位置特异性CATE |
| subgroup_profiles.csv | 亚群画像 |
| counterfactual_suggestions.csv | 反事实编辑建议 |
| truncation_suggestions.csv | 截短建议 |
| design_strategy.json | 机器可读设计策略 |
| design_strategy.txt | Proteo-R1可读策略 |
| csco_analysis_report.txt | 综合分析报告 |
| esm2_embeddings.npy | ESM-2嵌入 |

### 4.5 服务器部署
使用时需将脚本中的 REDACTED_IP 和 REDACTED_USER 替换为实际服务器地址和用户名。

## 五、变更记录 (Changelog)

### v2.0 (2026-06-02)
- 修复: _double_ml_cate 函数卡死问题（嵌套并行死锁、计算量爆炸、特征维度爆炸）
- 修复: _run_multi_treatment_cate 函数无响应问题
- 优化: CausalForestDML 参数调优（n_estimators 1000→200, cv 5→3, inner_n_jobs=1）
- 优化: backbone_id 编码从 one-hot 改为 category.cat.codes（避免维度爆炸）
- 优化: 数据预转换为 float32 + ascontiguousarray（加速矩阵运算）
- 新增: 每个 treatment 变量的独立计时和进度输出
- 新增: 错误隔离机制（单个 treatment 失败不阻塞）
- 新增: csco_mathematical_methods.md 数学方法总结文档
- 新增: csco_optimization_report.md 优化分析报告
- 安全: 全面数据脱敏处理
- 安全: .gitignore 新增排除规则

### v1.0 (2026-05-28)
- 初始版本
- 6阶段管线完整实现
- Cox PH + KM + PC算法 + ATE + Double ML + ESM-2 + 反事实导航

## 六、已知限制
1. econml 未安装时自动降级到 R-learner，CATE 精度略低
2. ESM-2 编码阶段耗时较长（10572序列约30-60分钟）
3. R-learner 的 multi_treatment_cate 阶段约53分钟（8个变量串行）
4. AF3 阶段95%失败率的原因尚未完全分析
5. 位置特异性CATE仅发现3个显著结果（Pos9 E, Pos6 R, Pos0 R），覆盖有限

## 七、致谢
- 数据来源: Q02223靶点抗体VH序列评估
- 工具: RF2, AF3, Schrodinger, Desmond, ESM-2
- 因果推断: lifelines, econml, sklearn, lightgbm
