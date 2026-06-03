#!/usr/bin/env python3
"""
CSC-O 公共配置与工具模块
统一管理常量、配置、特征提取函数，消除跨文件代码克隆
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AROMATIC = set('YWF')
POSITIVE = set('KRH')
NEGATIVE = set('DE')
HYDROPHOBIC = set('AVILMFWP')
GLYCINE = set('G')
SERINE = set('S')
PROLINE = set('P')
POLAR = set('STNQ')

ESM2_MODEL_REGISTRY = {
    "esm2_t12_35M_UR50D":  {"layers": 12, "dim": 480},
    "esm2_t30_150M_UR50D": {"layers": 30, "dim": 640},
    "esm2_t33_650M_UR50D": {"layers": 33, "dim": 1280},
    "esm2_t36_3B_UR50D":   {"layers": 36, "dim": 2560},
}

DEFAULT_CONFIG = {
    "input_file": None,
    "output_dir": "./output",
    "work_dir": "./work",
    "stage_resume": True,
    "esm2_model": "esm2_t12_35M_UR50D",
    "esm2_models": ["esm2_t12_35M_UR50D"],
    "esm2_fusion": "concat",
    "esm2_pca_dim": 256,
    "esm2_batch_size": 4,
    "cate_method": "causal_forest",
    "subgroup_n_clusters": 3,
    "device": "auto",
    "max_cdr3_len": 13,
    "truncation_target": 7,
    "counterfactual_top_n": 2000,
    "tsne_sample_size": 3000,
    "optimization_target": "final_candidate",
    "rf2_lddt_threshold": 0.86,
    "rf2_pae_threshold": 10.0,
    "rf2_rmsd_threshold": 2.5,
    "screener_top_n": 500,
    "screener_min_edit_distance": 2,
    "screener_prob_threshold": 0.2,
    "generator_n_samples": 20000,
    "generator_min_soft_score": 1.5,
}

STAGE_NAMES = [
    "data_engineering",
    "layer1_stratified",
    "layer2_causal",
    "layer3_esm2_encode",
    "layer3_counterfactual",
    "layer5_synthesis",
]

TREATMENT_VARS = [
    'cdr3_len', 'positive_ratio', 'aromatic_ratio',
    'glycine_ratio', 'serine_ratio', 'proline_count',
]

TREATMENT_VARS_EXTENDED = TREATMENT_VARS + ['first_is_aromatic', 'last_is_YH']

CONFOUNDER_COLS = ['backbone_id', 'aromatic_ratio', 'hydrophobic_ratio', 'positive_ratio']

CDR3_FEATURE_COLS = [
    'cdr3_len', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio',
    'proline_count', 'positive_count', 'hydrophobic_ratio',
    'first_is_aromatic', 'last_is_YH', 'has_ggg', 'has_sss', 'has_ll',
]

COLUMN_SCHEMA = {
    'global_sequence_index':        {'dtype': int,    'default': 0},
    'run':                          {'dtype': str,    'default': ''},
    'run_index':                    {'dtype': int,    'default': 0},
    'target_id':                    {'dtype': str,    'default': ''},
    'gene_symbol':                  {'dtype': str,    'default': ''},
    'design_id':                    {'dtype': str,    'default': ''},
    'backbone_id':                  {'dtype': int,    'default': 0},
    'sequence_variant_id':          {'dtype': int,    'default': 0},
    'vh_sequence':                  {'dtype': str,    'default': ''},
    'vl_sequence':                  {'dtype': str,    'default': ''},
    'full_sequence':                {'dtype': str,    'default': ''},
    'sequence_length':              {'dtype': int,    'default': 0},
    'cdr1_sequence':                {'dtype': str,    'default': ''},
    'cdr2_sequence':                {'dtype': str,    'default': ''},
    'cdr3_sequence':                {'dtype': str,    'default': ''},
    'framework_type':               {'dtype': str,    'default': ''},
    'hotspot_indices':              {'dtype': str,    'default': ''},
    'hotspots':                     {'dtype': str,    'default': ''},
    'hotspot_strategy':             {'dtype': str,    'default': ''},
    'hotspot_policy_mode':          {'dtype': str,    'default': ''},
    'hotspot_policy_default_role':  {'dtype': str,    'default': ''},
    'hotspot_policy_contact_cutoff': {'dtype': float, 'default': np.nan},
    'hotspot_policy_min_contacts':  {'dtype': float,  'default': np.nan},
    'hotspot_policy_min_coverage':  {'dtype': float,  'default': np.nan},
    'hotspot_index':                {'dtype': float,  'default': np.nan},
    'hotspot_run_hotspots':         {'dtype': str,    'default': ''},
    'hotspot_role':                 {'dtype': str,    'default': ''},
    'hotspot_confidence':           {'dtype': float,  'default': np.nan},
    'designed_hotspots':            {'dtype': str,    'default': ''},
    'contacted_hotspots':           {'dtype': str,    'default': ''},
    'rf2_pred_lddt':                {'dtype': float,  'default': np.nan},
    'rf2_pae':                      {'dtype': float,  'default': np.nan},
    'rf2_interaction_pae':          {'dtype': float,  'default': np.nan},
    'rf2_target_aligned_antibody_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_target_aligned_cdr_rmsd': {'dtype': float,  'default': np.nan},
    'rf2_framework_aligned_antibody_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_cdr_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_H1_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_H2_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_H3_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_L1_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_L2_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_framework_aligned_L3_rmsd': {'dtype': float, 'default': np.nan},
    'rf2_passed_filter':            {'dtype': bool,   'default': False},
    'rf2_filter_reason':            {'dtype': str,    'default': ''},
    'af3_analyzed':                 {'dtype': bool,   'default': False},
    'af3_plddt':                    {'dtype': float,  'default': np.nan},
    'af3_ptm':                      {'dtype': float,  'default': np.nan},
    'af3_iptm':                     {'dtype': float,  'default': np.nan},
    'af3_ranking_confidence':       {'dtype': float,  'default': np.nan},
    'af3_ranking_score':            {'dtype': float,  'default': np.nan},
    'af3_antibody_antigen_iptm':    {'dtype': float,  'default': np.nan},
    'af3_antibody_antigen_pae_min': {'dtype': float,  'default': np.nan},
    'af3_pae_interface_mean':       {'dtype': float,  'default': np.nan},
    'af3_passed_filter':            {'dtype': bool,   'default': False},
    'af3_filter_reason':            {'dtype': str,    'default': ''},
    'af_seed':                      {'dtype': float,  'default': np.nan},
    'schrodinger_analyzed':         {'dtype': bool,   'default': False},
    'schrodinger_passed_filter':    {'dtype': bool,   'default': False},
    'schrodinger_filter_reason':    {'dtype': str,    'default': ''},
    'docking_score':                {'dtype': float,  'default': np.nan},
    'mmgbsa_delta_g':               {'dtype': float,  'default': np.nan},
    'interface_area':               {'dtype': float,  'default': np.nan},
    'n_hydrogen_bonds':             {'dtype': float,  'default': np.nan},
    'n_salt_bridges':               {'dtype': float,  'default': np.nan},
    'n_hydrophobic_contacts':       {'dtype': float,  'default': np.nan},
    'n_clashes':                    {'dtype': float,  'default': np.nan},
    'hotspot_coverage':             {'dtype': float,  'default': np.nan},
    'cdr_contact_count':            {'dtype': float,  'default': np.nan},
    'pose_rmsd_to_alphafold':       {'dtype': float,  'default': np.nan},
    'desmond_rmsd':                 {'dtype': float,  'default': np.nan},
    'desmond_rmsf_mean':            {'dtype': float,  'default': np.nan},
    'desmond_interface_hbond_retention': {'dtype': float, 'default': np.nan},
    'desmond_hotspot_contact_retention': {'dtype': float, 'default': np.nan},
    'desmond_complex_dissociated':  {'dtype': object, 'default': np.nan},
    'funnel_stage':                 {'dtype': str,    'default': ''},
    'final_candidate':              {'dtype': bool,   'default': False},
    'antibody_pdb_path':            {'dtype': str,    'default': ''},
    'complex_pdb_path':             {'dtype': str,    'default': ''},
    'schrodinger_top_pose_pdb':     {'dtype': str,    'default': ''},
    'metrics_path':                 {'dtype': str,    'default': ''},
}

COLUMN_ALIASES = {
    'vh_sequence':  ['vh_sequence', 'vh_seq', 'heavy_chain', 'vh'],
    'full_sequence': ['full_sequence', 'sequence', 'seq', 'protein_sequence'],
    'cdr3_sequence': ['cdr3_sequence', 'cdr3', 'cdr3_seq', 'h3_sequence'],
    'cdr2_sequence': ['cdr2_sequence', 'cdr2', 'cdr2_seq', 'h2_sequence'],
    'cdr1_sequence': ['cdr1_sequence', 'cdr1', 'cdr1_seq', 'h1_sequence'],
    'rf2_pred_lddt': ['rf2_pred_lddt', 'pred_lddt', 'plddt'],
    'rf2_interaction_pae': ['rf2_interaction_pae', 'interaction_pae', 'pae'],
    'rf2_framework_aligned_cdr_rmsd': ['rf2_framework_aligned_cdr_rmsd', 'framework_aligned_cdr_rmsd', 'cdr_rmsd'],
    'rf2_passed_filter': ['rf2_passed_filter', 'passed_filter', 'rf2_pass', 'rf2_pass_filter'],
    'rf2_filter_reason': ['rf2_filter_reason', 'filter_reason'],
    'backbone_id': ['backbone_id', 'backbone'],
    'design_id': ['design_id', 'design'],
    'target_id': ['target_id', 'target'],
    'final_candidate': ['final_candidate', 'is_candidate', 'selected'],
    'af3_analyzed': ['af3_analyzed', 'af3_was_analyzed'],
    'schrodinger_analyzed': ['schrodinger_analyzed', 'schrodinger_was_analyzed'],
    'af3_passed_filter': ['af3_passed_filter', 'af3_pass'],
    'schrodinger_passed_filter': ['schrodinger_passed_filter', 'schrodinger_pass'],
    'funnel_stage': ['funnel_stage', 'stage'],
}


def extract_cdr3_features(cdr3_seq: str) -> Dict[str, Any]:
    if pd.isna(cdr3_seq) or len(str(cdr3_seq)) == 0:
        return {
            'cdr3_len': 0, 'positive_count': 0, 'positive_ratio': 0.0,
            'aromatic_count': 0, 'aromatic_ratio': 0.0,
            'glycine_count': 0, 'glycine_ratio': 0.0,
            'serine_count': 0, 'serine_ratio': 0.0,
            'proline_count': 0, 'proline_ratio': 0.0,
            'hydrophobic_ratio': 0.0, 'negative_ratio': 0.0,
            'first_residue': 'X', 'last_residue': 'X',
            'first_is_aromatic': 0, 'last_is_YH': 0,
            'has_ggg': 0, 'has_sss': 0, 'has_ll': 0,
        }
    s = str(cdr3_seq)
    n = len(s)
    first_aa = s[0]
    last_aa = s[-1]
    return {
        'cdr3_len': n,
        'positive_count': sum(1 for a in s if a in POSITIVE),
        'positive_ratio': sum(1 for a in s if a in POSITIVE) / n,
        'aromatic_count': sum(1 for a in s if a in AROMATIC),
        'aromatic_ratio': sum(1 for a in s if a in AROMATIC) / n,
        'glycine_count': sum(1 for a in s if a in GLYCINE),
        'glycine_ratio': sum(1 for a in s if a in GLYCINE) / n,
        'serine_count': sum(1 for a in s if a in SERINE),
        'serine_ratio': sum(1 for a in s if a in SERINE) / n,
        'proline_count': sum(1 for a in s if a in PROLINE),
        'proline_ratio': sum(1 for a in s if a in PROLINE) / n,
        'hydrophobic_ratio': sum(1 for a in s if a in HYDROPHOBIC) / n,
        'negative_ratio': sum(1 for a in s if a in NEGATIVE) / n,
        'first_residue': first_aa,
        'last_residue': last_aa,
        'first_is_aromatic': int(first_aa in AROMATIC),
        'last_is_YH': int(last_aa in ('Y', 'H')),
        'has_ggg': int('GGG' in s),
        'has_sss': int('SSS' in s.upper()),
        'has_ll': int('LL' in s),
    }


def get_optimization_target(feat_df: pd.DataFrame, config: Dict) -> tuple:
    target = config.get('optimization_target', 'final_candidate')
    if target == 'final_candidate' and 'final_candidate' in feat_df.columns:
        n_pos = feat_df['final_candidate'].sum()
        if n_pos >= 10:
            y_binary = feat_df['final_candidate'].astype(int).values
            y_continuous = feat_df['rf2_interaction_pae'].values.astype(float)
            target_label = 'final_candidate'
            print(f"  优化目标: final_candidate (正样本数={n_pos}, 比例={n_pos/len(feat_df)*100:.2f}%)")
            return y_binary, y_continuous, target_label
        else:
            print(f"  警告: final_candidate正样本不足({n_pos}<10), 降级到rf2_passed")
    y_binary = feat_df['rf2_passed'].astype(int).values
    y_continuous = feat_df['rf2_interaction_pae'].values.astype(float)
    target_label = 'rf2_passed'
    print(f"  优化目标: rf2_passed (正样本数={y_binary.sum()}, 比例={y_binary.mean()*100:.1f}%)")
    return y_binary, y_continuous, target_label
