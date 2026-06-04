#!/usr/bin/env python3
"""第一批测试: Phase 1+2 集成测试"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载数据
df = pd.read_csv('output_server_v2.3/feature_matrix.csv')
emb = np.load('output_server_v2.3/esm2_embeddings.npy')

print('=== Phase 1: 多阶段中介模型 ===')
from csco_multistage_causal import MultiStageMediationModel

model = MultiStageMediationModel(
    available_stages=['rf2', 'final'],
    method='decomposition',
    verbose=True,
)

results = model.fit(
    df=df,
    embeddings=emb,
    treatment_cols=['first_is_aromatic', 'cdr3_length_bin', 'glycine_ratio_bin', 'serine_ratio_bin'],
    confounder_cols=['backbone_id'],
)

os.makedirs('output_v3_funnel', exist_ok=True)
model.save_results('output_v3_funnel/mediation_effects.csv')

print('\n\n=== Phase 2: 多状态生存模型 ===')
from csco_multistate_survival import MultiStateSurvivalModel

surv_model = MultiStateSurvivalModel(
    available_stages=['rf2', 'final'],
    verbose=True,
)

surv_results = surv_model.fit(
    df=df,
    treatment_cols=['first_is_aromatic', 'cdr3_length_bin', 'glycine_ratio_bin', 'serine_ratio_bin'],
    confounder_cols=['backbone_id'],
)

surv_model.save_results('output_v3_funnel/multistate_hazard_ratios.csv')

# 交叉验证
print('\n\n=== 交叉验证: ATE方向 vs Cox HR ===')
cv_df = surv_model.cross_validate_with_mediation(results)
cv_df.to_csv('output_v3_funnel/cross_validation_mediation_survival.csv', index=False)
print(cv_df.to_string())

print('\n\n=== 汇总 ===')
med_df = model.get_results_dataframe()
print('\n中介效应结果:')
cols = [c for c in ['treatment', 'total_effect', 'total_effect_se', 'direct_effect', 'indirect_effect', 'interpretation'] if c in med_df.columns]
print(med_df[cols].to_string())

surv_df = surv_model.get_results_dataframe()
print('\n多状态HR结果:')
cols = [c for c in ['treatment', 'combined_hr', 'rf2_hr', 'final_hr', 'dominant_stage', 'interpretation'] if c in surv_df.columns]
print(surv_df[cols].to_string())
