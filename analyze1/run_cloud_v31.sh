#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CSC-O v3.1 云端部署脚本
# 目标: 8×1080Ti服务器, 跑完整pipeline + v3.1生成 + Top200筛选
# 预计耗时: ~30分钟
# ═══════════════════════════════════════════════════════════════

set -e

echo "=========================================="
echo "CSC-O v3.1 云端部署"
echo "=========================================="

# === 配置 ===
WORK_DIR=$(pwd)
DATA_DIR="${WORK_DIR}/output_server_v2.3"
OUTPUT_DIR="${WORK_DIR}/output_v3_cloud"
STRATEGY_V24="${WORK_DIR}/output_v2.4_test/design_strategy_v2.4.json"
STRATEGY_V31="${WORK_DIR}/output_v3_funnel/design_strategy_v3.0.json"

mkdir -p "${OUTPUT_DIR}"

# === Step 1: 环境检查 ===
echo ""
echo "[Step 1/6] 环境检查..."
python3 -c "import numpy, pandas, sklearn, lifelines; print('  核心依赖OK')" || {
    echo "  安装依赖..."
    pip install numpy pandas scikit-learn lifelines esm torch
}

# 检查ESM-2模型
python3 -c "import esm; model, alphabet = esm.pretrained.esm2_t12_35M_UR50D(); print('  ESM-2 t12 OK')" || {
    echo "  [WARN] ESM-2模型未缓存, 首次运行将自动下载"
}

# === Step 2: ESM-2编码 ===
echo ""
echo "[Step 2/6] ESM-2编码 (t12+t30 fusion)..."
if [ -f "${DATA_DIR}/esm2_embeddings.npy" ]; then
    echo "  已有ESM-2嵌入, 跳过"
else
    echo "  运行ESM-2编码..."
    python3 csco_esm2_encode.py \
        --input "${DATA_DIR}/feature_matrix.csv" \
        --output "${DATA_DIR}" \
        --models t12 t30
fi

# === Step 3: 完整Pipeline (v3.1) ===
echo ""
echo "[Step 3/6] 运行v3.1 Pipeline..."
python3 csco_pipeline.py \
    --input "${DATA_DIR}/feature_matrix.csv" \
    --output "${OUTPUT_DIR}/pipeline" \
    --target final_candidate \
    --esm2-models t12+t30 \
    || echo "  [WARN] Pipeline部分stage可能失败, 继续..."

# === Step 4: 概率校准 ===
echo ""
echo "[Step 4/6] 概率校准..."
python3 csco_probability_calibration.py \
    --data "${DATA_DIR}/feature_matrix.csv" \
    --output "${OUTPUT_DIR}"

# === Step 5: v2.4 vs v3.1 对比生成 ===
echo ""
echo "[Step 5/6] 生成序列 (v2.4 vs v3.1)..."

# v2.4
echo "  生成v2.4序列..."
python3 -c "
from csco_generator import load_strategy, generate_cdr3
from csco_screener import score_sequences
import random, json
rng = random.Random(42)
strategy = load_strategy('${STRATEGY_V24}')
seqs = []
for l in strategy['hard_constraints']['cdr3_length_allowed']:
    seqs.extend(generate_cdr3(strategy, l, 5000, rng))
with open('${OUTPUT_DIR}/v24_raw.json', 'w') as f:
    json.dump([s['cdr3'] for s in seqs], f)
print(f'  v2.4: {len(seqs)} 条')
"

# v3.1 (校准后)
echo "  生成v3.1序列 (校准后)..."
python3 -c "
import sys, json, numpy as np
sys.path.insert(0, '${WORK_DIR}')
from csco_funnel_aware_strategy import FunnelAwareStrategy
from csco_funnel_generator import FunnelAwareGenerator
from csco_probability_calibration import ProbabilityCalibrator

# 加载校准器
strategy = FunnelAwareStrategy(verbose=False)
cal = ProbabilityCalibrator(verbose=False)
cal.fit_from_dataframe(
    __import__('pandas').read_csv('${DATA_DIR}/feature_matrix.csv'),
    strategy
)

# 生成
gen = FunnelAwareGenerator(
    strategy=strategy,
    base_strategy_path='${STRATEGY_V31}',
    calibrator=cal,
    min_edit_distance=3,
    verbose=True,
)
seqs = gen.generate(n_samples=10000, top_n=500, seed=42)
df = gen.generate_to_csv('${OUTPUT_DIR}/v31_calibrated_sequences.csv', n_samples=10000, top_n=500, seed=42)
print(f'  v3.1: {len(seqs)} 条')
print(f'  校准后P(final)均值: {np.mean([s.estimated_p_final for s in seqs]):.6f}')
"

# === Step 6: Top 200筛选 ===
echo ""
echo "[Step 6/6] Top 200筛选..."
python3 -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/v31_calibrated_sequences.csv')
if 'calibrated_p_final' in df.columns:
    top = df.nlargest(200, 'calibrated_p_final')
else:
    top = df.nlargest(200, 'predicted_p_final')
top.to_csv('${OUTPUT_DIR}/top200_candidates.csv', index=False)
print(f'  Top 200: P(final)范围 [{top.iloc[-1].get(\"calibrated_p_final\", top.iloc[-1].get(\"predicted_p_final\", 0)):.6f}, {top.iloc[0].get(\"calibrated_p_final\", top.iloc[0].get(\"predicted_p_final\", 0)):.6f}]')
print(f'  首残基分布: {dict(top[\"first_aa\"].value_counts().head(5))}')
"

# === 完成 ===
echo ""
echo "=========================================="
echo "云端部署完成!"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  ${OUTPUT_DIR}/v31_calibrated_sequences.csv  (500条校准后序列)"
echo "  ${OUTPUT_DIR}/top200_candidates.csv          (Top 200候选)"
echo "  ${OUTPUT_DIR}/calibration_params.json         (校准参数)"
echo "  ${OUTPUT_DIR}/pipeline/                       (Pipeline输出)"
echo ""
echo "下一步: 将top200_candidates.csv送实验验证"
echo "  结果回流: python3 csco_validation_framework.py --experimental results.csv"
