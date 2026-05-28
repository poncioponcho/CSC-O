#!/usr/bin/env python3
"""
CSC-O Pipeline — Causal-Stratified Counterfactual Optimization
统一入口：自动检测输入格式、补全缺失列、顺序执行全部6个阶段
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "input_file": None,
    "output_dir": "./output",
    "work_dir": "./work",
    "stage_resume": True,
    "esm2_model": "esm2_t12_35M_UR50D",
    "esm2_batch_size": 4,
    "device": "auto",
    "max_cdr3_len": 13,
    "truncation_target": 7,
    "counterfactual_top_n": 2000,
    "tsne_sample_size": 3000,
}

STAGE_NAMES = [
    "data_engineering",
    "layer1_stratified",
    "layer2_causal",
    "layer3_esm2_encode",
    "layer3_counterfactual",
    "layer5_synthesis",
]

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AROMATIC = set('YWF')
POSITIVE = set('KRH')
NEGATIVE = set('DE')
HYDROPHOBIC = set('AVILMFWP')
GLYCINE = set('G')
SERINE = set('S')
PROLINE = set('P')

# ═══════════════════════════════════════════════════════════════
# 完整列模式：81列 + 默认值
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# 进度持久化
# ═══════════════════════════════════════════════════════════════

class ProgressTracker:
    def __init__(self, work_dir):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.work_dir / "pipeline_state.json"
        self.state = self._load()

    def _load(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"started_at": None, "completed_stages": [], "current_stage": None, "errors": []}

    def save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def is_completed(self, name):
        return name in [s["name"] for s in self.state["completed_stages"]]

    def start_stage(self, name):
        self.state["current_stage"] = name
        self.state["started_at"] = datetime.now().isoformat()
        self.save()
        print(f"\n{'='*60}\n>>> 启动阶段: {name}\n{'='*60}")
        return time.time()

    def complete_stage(self, name, t0, metrics=None):
        elapsed = time.time() - t0
        self.state["completed_stages"].append({
            "name": name,
            "completed_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 1),
            "metrics": metrics or {}
        })
        self.state["current_stage"] = None
        self.save()
        print(f"<<< 阶段完成: {name} | 耗时: {elapsed:.1f}s\n")

    def log_error(self, stage, error):
        self.state["errors"].append({
            "stage": stage,
            "time": datetime.now().isoformat(),
            "error": str(error),
            "traceback": traceback.format_exc()
        })
        self.save()

# ═══════════════════════════════════════════════════════════════
# 数据适配层
# ═══════════════════════════════════════════════════════════════

class DataAdapter:
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.df = self._load_file()
        self._normalize_columns()
        self._fill_missing_columns()
        self._coerce_types()
        self._derive_missing_fields()

    def _load_file(self):
        suffix = self.input_path.suffix.lower()
        if suffix in ('.xlsx', '.xls'):
            print(f"  检测到Excel格式: {self.input_path.name}")
            return pd.read_excel(self.input_path)
        elif suffix == '.csv':
            print(f"  检测到CSV格式: {self.input_path.name}")
            return pd.read_csv(self.input_path)
        elif suffix in ('.tsv', '.txt'):
            print(f"  检测到TSV格式: {self.input_path.name}")
            return pd.read_csv(self.input_path, sep='\t')
        else:
            try:
                return pd.read_csv(self.input_path)
            except Exception:
                return pd.read_excel(self.input_path)

    def _normalize_columns(self):
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        for standard, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in self.df.columns and standard not in self.df.columns:
                    self.df.rename(columns={alias: standard}, inplace=True)
                    break

    def _fill_missing_columns(self):
        present = set(self.df.columns)
        missing = []
        for col, spec in COLUMN_SCHEMA.items():
            if col not in present:
                self.df[col] = spec['default']
                missing.append(col)
        if missing:
            print(f"  补全缺失列 {len(missing)} 个: {missing[:10]}{'...' if len(missing)>10 else ''}")

    def _coerce_types(self):
        bool_cols = [c for c, s in COLUMN_SCHEMA.items() if s['dtype'] == bool]
        for col in bool_cols:
            if col in self.df.columns:
                raw = self.df[col]
                if raw.dtype in (np.float64, np.float32, np.int64, np.int32):
                    self.df[col] = raw.fillna(0).astype(bool)
                else:
                    self.df[col] = raw.astype(str).str.lower().map({
                        'true': True, '1': True, 'yes': True, '1.0': True,
                        'false': False, '0': False, 'no': False, '0.0': False,
                        'nan': False, 'none': False, '': False,
                    }).fillna(False)

    def _derive_missing_fields(self):
        if self._is_column_effectively_empty('vh_sequence'):
            if 'full_sequence' in self.df.columns and not self._is_column_effectively_empty('full_sequence'):
                self.df['vh_sequence'] = self.df['full_sequence']

        if self._is_column_effectively_empty('cdr3_sequence'):
            if 'vh_sequence' in self.df.columns and not self._is_column_effectively_empty('vh_sequence'):
                self.df['cdr3_sequence'] = self.df['vh_sequence'].apply(self._extract_cdr3_heuristic)

        if self._is_column_effectively_empty('funnel_stage'):
            self.df['funnel_stage'] = self.df.apply(self._infer_funnel_stage, axis=1)

        if self._is_column_effectively_empty('rf2_passed_filter', treat_all_false_as_empty=True):
            if 'rf2_pred_lddt' in self.df.columns and 'rf2_interaction_pae' in self.df.columns:
                lddt_ok = self.df['rf2_pred_lddt'] >= 0.88
                pae_ok = self.df['rf2_interaction_pae'] <= 10.0
                rmsd_ok = self.df['rf2_framework_aligned_cdr_rmsd'].fillna(0) <= 2.0
                self.df['rf2_passed_filter'] = lddt_ok & pae_ok & rmsd_ok

    def _is_column_effectively_empty(self, col_name, treat_all_false_as_empty=False):
        if col_name not in self.df.columns:
            return True
        col = self.df[col_name]
        if col.isna().all():
            return True
        if col.dtype == object and (col == '').all():
            return True
        if treat_all_false_as_empty and col.dtype == bool and (~col).all():
            return True
        return False

    @staticmethod
    def _extract_cdr3_heuristic(vh_seq):
        if pd.isna(vh_seq) or len(str(vh_seq)) < 20:
            return ""
        vh_seq = str(vh_seq)
        for motif in ['WGQGTLVTVS', 'WGQGTLVTV', 'WGQGTLVTVSS']:
            if motif in vh_seq:
                pos = vh_seq.find(motif)
                if pos > 15:
                    return vh_seq[pos-15:pos]
        return vh_seq[-20:-10] if len(vh_seq) > 30 else ""

    @staticmethod
    def _infer_funnel_stage(row):
        if row['final_candidate']:
            return 'final_candidate'
        if not row['rf2_passed_filter']:
            return 'rf2_failed'
        if row.get('schrodinger_passed_filter', False):
            return 'schrodinger_passed'
        if row.get('af3_passed_filter', False):
            return 'af3_passed'
        if row.get('schrodinger_analyzed', False):
            return 'schrodinger_failed'
        if row.get('af3_analyzed', False):
            return 'af3_failed'
        return 'rf2_passed'

    def validate(self):
        required = ['vh_sequence', 'rf2_pred_lddt', 'rf2_interaction_pae', 'rf2_passed_filter']
        missing = [c for c in required if c not in self.df.columns or self.df[c].isna().all()]
        if missing:
            print(f"  警告: 以下必需列全为空: {missing}")
        n_valid = self.df.dropna(subset=['rf2_pred_lddt', 'rf2_interaction_pae']).shape[0]
        if n_valid < 50:
            raise ValueError(f"有效数据不足 ({n_valid} 行), 需要至少50条有RF2指标的记录")
        return self.df

# ═══════════════════════════════════════════════════════════════
# CDR3特征提取
# ═══════════════════════════════════════════════════════════════

def extract_cdr3_features(cdr3_seq):
    if pd.isna(cdr3_seq) or len(str(cdr3_seq)) == 0:
        return {
            'cdr3_len': 0, 'positive_count': 0, 'positive_ratio': 0.0,
            'aromatic_count': 0, 'aromatic_ratio': 0.0,
            'glycine_count': 0, 'glycine_ratio': 0.0,
            'serine_count': 0, 'serine_ratio': 0.0,
            'proline_count': 0, 'proline_ratio': 0.0,
            'hydrophobic_ratio': 0.0, 'negative_ratio': 0.0,
            'first_residue': 'X', 'last_residue': 'X',
            'has_ggg': False, 'has_sss': False, 'has_ll': False,
        }
    s = str(cdr3_seq)
    n = len(s)
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
        'first_residue': s[0],
        'last_residue': s[-1],
        'has_ggg': 'GGG' in s,
        'has_sss': 'SSS' in s.upper(),
        'has_ll': 'LL' in s,
    }

# ═══════════════════════════════════════════════════════════════
# 阶段1: 数据工程
# ═══════════════════════════════════════════════════════════════

def stage_data_engineering(df, output_dir, config):
    out = Path(output_dir)
    print(f"  输入数据: {len(df)} 行, {len(df.columns)} 列")

    features_list = []
    for _, row in df.iterrows():
        feats = extract_cdr3_features(row['cdr3_sequence'])
        feats['global_sequence_index'] = row['global_sequence_index']
        feats['rf2_passed'] = row['rf2_passed_filter']
        feats['rf2_pred_lddt'] = row['rf2_pred_lddt']
        feats['rf2_interaction_pae'] = row['rf2_interaction_pae']
        feats['funnel_stage'] = row['funnel_stage']
        feats['backbone_id'] = row['backbone_id']
        feats['hotspot_strategy'] = row['hotspot_strategy']
        feats['framework_type'] = row['framework_type']
        feats['final_candidate'] = row['final_candidate']
        feats['cdr3_sequence'] = row['cdr3_sequence']
        feats['vh_sequence'] = row['vh_sequence']

        feats['survival_time'] = 1 if row['rf2_passed_filter'] else 0
        feats['survival_event'] = 0 if row['rf2_passed_filter'] else 1
        feats['death_stage'] = 0 if row['rf2_passed_filter'] else 1

        feats['af3_analyzed'] = row['af3_analyzed']
        feats['schrodinger_analyzed'] = row['schrodinger_analyzed']
        feats['af3_passed'] = row.get('af3_passed_filter', False)
        feats['schrodinger_passed'] = row.get('schrodinger_passed_filter', False)
        feats['af3_iptm'] = row.get('af3_iptm', np.nan)
        feats['mmgbsa_delta_g'] = row.get('mmgbsa_delta_g', np.nan)
        feats['rf2_filter_reason'] = row.get('rf2_filter_reason', '')

        features_list.append(feats)

    feat_df = pd.DataFrame(features_list)
    feat_df.to_csv(out / "feature_matrix.csv", index=False)

    survival_records = []
    for _, row in feat_df.iterrows():
        stage_times = {1: row['rf2_passed'], 2: row['af3_passed'],
                       3: row['schrodinger_passed'], 4: row['final_candidate']}
        if not stage_times[1]:
            t, event = 1, 1
        elif not stage_times[2]:
            t, event = 2, 1
        elif not stage_times[3]:
            t, event = 3, 1
        elif not stage_times[4]:
            t, event = 4, 1
        else:
            t, event = 4, 0
        survival_records.append({
            'global_sequence_index': row['global_sequence_index'],
            'time': t, 'event': event,
            'cdr3_len': row['cdr3_len'], 'positive_ratio': row['positive_ratio'],
            'aromatic_ratio': row['aromatic_ratio'], 'glycine_ratio': row['glycine_ratio'],
            'serine_ratio': row['serine_ratio'], 'proline_count': row['proline_count'],
            'hydrophobic_ratio': row['hydrophobic_ratio'], 'negative_ratio': row['negative_ratio'],
            'first_residue': row['first_residue'], 'last_residue': row['last_residue'],
            'backbone_id': row['backbone_id'],
            'rf2_interaction_pae': row['rf2_interaction_pae'], 'rf2_pred_lddt': row['rf2_pred_lddt'],
        })

    surv_df = pd.DataFrame(survival_records)
    surv_df.to_csv(out / "survival_data.csv", index=False)

    print(f"  Feature matrix: {feat_df.shape}, Survival data: {surv_df.shape}")
    return {"n_samples": len(df), "n_features": len(feat_df.columns)}

# ═══════════════════════════════════════════════════════════════
# 阶段2: 分层归因与动态阈值校准
# ═══════════════════════════════════════════════════════════════

def stage_layer1_stratified(output_dir, config):
    from lifelines import CoxPHFitter, KaplanMeierFitter
    out = Path(output_dir)
    surv_df = pd.read_csv(out / "survival_data.csv")
    feat_df = pd.read_csv(out / "feature_matrix.csv")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    total = len(surv_df)
    s1 = (surv_df['time'] == 1) & (surv_df['event'] == 1)
    s2 = (surv_df['time'] == 2) & (surv_df['event'] == 1)
    s3 = (surv_df['time'] == 3) & (surv_df['event'] == 1)
    s4 = (surv_df['time'] == 4) & (surv_df['event'] == 1)
    sv = surv_df['event'] == 0
    print(f"  RF2失败: {s1.sum()} ({s1.sum()/total*100:.1f}%), AF3失败: {s2.sum()}, 最终候选: {sv.sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    stages = ['RF2', 'AF3', 'Schrödinger', 'Desmond', 'Candidate']
    counts = [s1.sum(), s2.sum(), s3.sum(), s4.sum(), sv.sum()]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
    axes[0].bar(stages, counts, color=colors)
    axes[0].set_title('Attrition by Pipeline Stage')
    axes[0].set_ylabel('Sequences')
    for i, v in enumerate(counts):
        axes[0].text(i, v + 50, str(v), ha='center')

    cum_stages = ['Start'] + stages
    cumulative = [total]
    rem = total
    for c in counts:
        rem -= c
        cumulative.append(max(rem, 0))
    axes[1].plot(cum_stages, cumulative, 'o-', color='#2c3e50', linewidth=2, markersize=8)
    axes[1].fill_between(range(len(cum_stages)), cumulative, alpha=0.15, color='#3498db')
    axes[1].set_title('Cumulative Survival')
    axes[1].set_ylabel('Remaining')
    plt.tight_layout()
    plt.savefig(out / 'layer1_attrition_funnel.png', dpi=150, bbox_inches='tight')
    plt.close()

    cox_df = surv_df.copy()
    for v in ['cdr3_len', 'positive_ratio', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'hydrophobic_ratio']:
        cox_df[f'{v}_centered'] = cox_df[v] - cox_df[v].mean()
    cox_vars = [f'{v}_centered' for v in ['cdr3_len', 'positive_ratio', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'hydrophobic_ratio']]
    cox_data = cox_df[['time', 'event'] + cox_vars].dropna()
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='time', event_col='event')
    cph.print_summary()

    hr_results = []
    for var in cox_vars:
        coef = cph.params_[var]
        hr = np.exp(coef)
        ci_l = np.exp(cph.confidence_intervals_.loc[var, '95% lower-bound'])
        ci_u = np.exp(cph.confidence_intervals_.loc[var, '95% upper-bound'])
        p_val = cph.summary.loc[var, 'p']
        hr_results.append({'variable': var.replace('_centered', ''), 'coef': coef, 'HR': hr, 'CI_lower': ci_l, 'CI_upper': ci_u, 'p_value': p_val})
    hr_df = pd.DataFrame(hr_results).sort_values('HR', ascending=False)
    hr_df.to_csv(out / 'cox_hazard_ratios.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(hr_df))
    ax.errorbar(hr_df['HR'], y_pos, xerr=[hr_df['HR'] - hr_df['CI_lower'], hr_df['CI_upper'] - hr_df['HR']], fmt='o', capsize=4)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hr_df['variable'])
    ax.set_xlabel('Hazard Ratio (95% CI)')
    ax.set_title('Cox PH Hazard Ratios')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(out / 'layer1_forest_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    for label, mask in {'5-7': surv_df['cdr3_len'].isin([5,6,7]), '8-9': surv_df['cdr3_len'].isin([8,9]), '10+': surv_df['cdr3_len']>=10}.items():
        kmf.fit(surv_df.loc[mask, 'time'], surv_df.loc[mask, 'event'], label=label)
        kmf.plot_survival_function(ax=ax)
    ax.set_title('KM Survival by CDR3 Length')
    plt.tight_layout()
    plt.savefig(out / 'layer1_km_by_cdr3_length.png', dpi=150, bbox_inches='tight')
    plt.close()

    lddt_values = feat_df['rf2_pred_lddt'].values.astype(float)
    pae_values = feat_df['rf2_interaction_pae'].values.astype(float)
    final_candidates = feat_df['final_candidate'].values.astype(bool)
    sens_results = []
    for thresh in np.arange(0.82, 0.92, 0.005):
        pred_pass = (lddt_values >= thresh) & (pae_values <= 10.0)
        tp = np.sum(pred_pass & final_candidates)
        fp = np.sum(pred_pass & ~final_candidates)
        fn = np.sum(~pred_pass & final_candidates)
        tn = np.sum(~pred_pass & ~final_candidates)
        sens_results.append({
            'threshold': thresh, 'pass_rate': pred_pass.sum() / len(pred_pass),
            'sensitivity': tp / max(tp + fn, 1), 'precision': tp / max(tp + fp, 1),
        })
    pd.DataFrame(sens_results).to_csv(out / 'threshold_sensitivity.csv', index=False)

    len_pass = feat_df.groupby('cdr3_len').agg(total=('rf2_passed', 'count'), rf2_pass=('rf2_passed', 'sum'), final_cand=('final_candidate', 'sum')).reset_index()
    len_pass['rf2_pass_rate'] = len_pass['rf2_pass'] / len_pass['total'] * 100
    len_pass['candidate_rate'] = len_pass['final_cand'] / len_pass['total'] * 100

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(len_pass['cdr3_len'] - 0.2, len_pass['rf2_pass_rate'], 0.4, label='RF2 Pass %', color='#3498db')
    ax1.bar(len_pass['cdr3_len'] + 0.2, len_pass['candidate_rate'], 0.4, label='Candidate %', color='#2ecc71')
    ax1.set_xlabel('CDR3 Length'); ax1.set_ylabel('Rate (%)'); ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(len_pass['cdr3_len'], len_pass['total'], 'r--o')
    ax2.set_ylabel('Count', color='r')
    plt.tight_layout()
    plt.savefig(out / 'layer1_cdr3_length_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {"cox_concordance": cph.concordance_index_}

# ═══════════════════════════════════════════════════════════════
# 阶段3: 因果约束引擎
# ═══════════════════════════════════════════════════════════════

def stage_layer2_causal(output_dir, config):
    from scipy import stats
    from itertools import combinations
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_predict
    out = Path(output_dir)
    feat_df = pd.read_csv(out / "feature_matrix.csv")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    treatment_vars = ['cdr3_len', 'positive_ratio', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'proline_count']
    outcome_binary = 'rf2_passed'
    outcome_continuous = 'rf2_interaction_pae'

    feat_df['first_is_aromatic'] = feat_df['first_residue'].isin(['Y', 'W', 'F']).astype(int)
    feat_df['last_is_YH'] = feat_df['last_residue'].isin(['Y', 'H']).astype(int)
    treatment_vars_extended = treatment_vars + ['first_is_aromatic', 'last_is_YH']

    causal_cols = treatment_vars_extended + [outcome_binary, outcome_continuous, 'backbone_id']
    causal_data = feat_df[causal_cols].copy()
    causal_data[outcome_binary] = causal_data[outcome_binary].astype(int)

    def partial_corr_test(x, y, z_data, alpha=0.05):
        if z_data.shape[1] == 0:
            return stats.pearsonr(x, y)
        lr = LinearRegression()
        lr.fit(z_data, x); res_x = x - lr.predict(z_data)
        lr.fit(z_data, y); res_y = y - lr.predict(z_data)
        return stats.pearsonr(res_x, res_y)

    def pc_algorithm(data, alpha=0.01, domain_constraints=None):
        nodes = list(data.columns); n = len(nodes)
        adj = {i: set(range(n)) - {i} for i in range(n)}
        sep_sets = {}
        if domain_constraints is None: domain_constraints = []
        depth = 0
        while depth <= 3:
            removed = []
            for i in range(n):
                for j in list(adj[i]):
                    if j not in adj[i]: continue
                    neighbors = adj[i] - {j}
                    if len(neighbors) < depth: continue
                    for cond_set in combinations(neighbors, depth):
                        cond_vars = [nodes[k] for k in cond_set]
                        z_data = data[cond_vars].values if cond_vars else np.empty((len(data), 0))
                        _, p = partial_corr_test(data[nodes[i]].values, data[nodes[j]].values, z_data)
                        if p > alpha:
                            adj[i].discard(j); adj[j].discard(i)
                            sep_sets[(i,j)] = cond_set; sep_sets[(j,i)] = cond_set
                            removed.append((i,j)); break
            if not removed: break
            depth += 1
        dag = {i: set() for i in range(n)}
        for i in range(n):
            for j in adj[i]:
                if j in adj[i] and i in adj[j]:
                    is_constrained = any(nodes[j]==src and nodes[i]==dst for src,dst in domain_constraints)
                    if is_constrained: dag[j].add(i)
                    elif (i,j) in sep_sets: dag[i].add(j)
                    elif (j,i) in sep_sets: dag[j].add(i)
                    elif abs(data[nodes[i]].corr(data[nodes[j]])) > 0: dag[i].add(j)
                elif j in adj[i]: dag[i].add(j)
        return dag, nodes, sep_sets

    domain_constraints = [(o, t) for o in [outcome_binary, outcome_continuous] for t in treatment_vars_extended]
    dag, nodes, sep_sets = pc_algorithm(causal_data, alpha=0.01, domain_constraints=domain_constraints)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = {}
    for i, node in enumerate(treatment_vars[:6]): pos[node] = (i*2, 2)
    for i, node in enumerate(['first_is_aromatic', 'last_is_YH']): pos[node] = (i*2+3, 1)
    for i, node in enumerate([outcome_binary, outcome_continuous]): pos[node] = (i*2+4, 0)
    pos['backbone_id'] = (0, 0.5)
    for node_name, (x, y) in pos.items():
        if node_name in nodes:
            color = '#e74c3c' if node_name in [outcome_binary, outcome_continuous] else '#3498db' if node_name in treatment_vars_extended else '#95a5a6'
            ax.scatter(x, y, s=2000, c=color, zorder=5, alpha=0.8)
            ax.text(x, y, node_name, ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    for src in range(len(nodes)):
        for dst in dag[src]:
            if nodes[src] in pos and nodes[dst] in pos:
                x1,y1 = pos[nodes[src]]; x2,y2 = pos[nodes[dst]]
                ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5, alpha=0.6))
    ax.set_title('Learned Causal DAG'); ax.axis('off')
    plt.tight_layout()
    plt.savefig(out / 'layer2_causal_dag.png', dpi=150, bbox_inches='tight')
    plt.close()

    def backdoor_ate(data, treatment, outcome, confounders, binary_treatment=False):
        if binary_treatment:
            X = data[confounders].values; T = data[treatment].values; Y = data[outcome].values
            lr = LogisticRegression(max_iter=1000)
            ps = cross_val_predict(lr, X, T, cv=5, method='predict_proba')[:, 1]
            ps = np.clip(ps, 0.01, 0.99)
            ipw = np.where(T == 1, 1.0/ps, 1.0/(1-ps)); ipw = np.clip(ipw, 0, 10)
            return np.mean(Y * T * ipw - Y * (1-T) * ipw) / np.mean(ipw), None
        else:
            X_confounders = data[confounders].values; T = data[treatment].values.reshape(-1,1); Y = data[outcome].values
            X_full = np.hstack([T, X_confounders])
            lr = LinearRegression(); lr.fit(X_full, Y); ate = lr.coef_[0]
            n = len(Y); k = X_full.shape[1]
            residuals = Y - lr.predict(X_full); mse = np.sum(residuals**2) / (n-k-1)
            try:
                inv_diag = np.linalg.inv(X_full.T @ X_full)[0,0]
            except np.linalg.LinAlgError:
                inv_diag = np.linalg.pinv(X_full.T @ X_full)[0,0]
            se = np.sqrt(mse * inv_diag)
            t_stat = ate / se; p_value = 2*(1 - stats.t.cdf(abs(t_stat), df=n-k-1))
            return ate, (se, t_stat, p_value, ate - 1.96*se, ate + 1.96*se)

    ate_results = []
    for tv in treatment_vars_extended:
        confounders = ['backbone_id']
        if tv in ['cdr3_len', 'proline_count']:
            feat_df[f'{tv}_binary'] = (feat_df[tv] > feat_df[tv].median()).astype(int)
            ate, _ = backdoor_ate(feat_df, f'{tv}_binary', outcome_binary, confounders, binary_treatment=True)
            ate_results.append({'treatment': tv, 'outcome': outcome_binary, 'ATE': ate, 'SE': 0, 't_stat': 0, 'p_value': 0, 'CI_lower': 0, 'CI_upper': 0})
        ate_cont, info = backdoor_ate(feat_df, tv, outcome_continuous, confounders)
        se, t_stat, p_val, ci_l, ci_u = info if info else (0,0,0,0,0)
        ate_results.append({'treatment': tv, 'outcome': outcome_continuous, 'ATE': ate_cont, 'SE': se, 't_stat': t_stat, 'p_value': p_val, 'CI_lower': ci_l, 'CI_upper': ci_u})
    ate_df = pd.DataFrame(ate_results)
    ate_df.to_csv(out / 'ate_estimates.csv', index=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    binary_ate = ate_df[ate_df['outcome'] == outcome_binary].sort_values('ATE', ascending=True)
    ax.barh(range(len(binary_ate)), binary_ate['ATE'], color=['#e74c3c' if p<0.05 else '#95a5a6' for p in binary_ate['p_value']], alpha=0.7, height=0.6)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_yticks(range(len(binary_ate))); ax.set_yticklabels(binary_ate['treatment'])
    ax.set_xlabel('ATE on RF2 Pass Rate'); ax.set_title('Causal ATE Estimates')
    plt.tight_layout()
    plt.savefig(out / 'layer2_ate_forest.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {"n_ate_results": len(ate_df)}

# ═══════════════════════════════════════════════════════════════
# 阶段4: ESM-2编码
# ═══════════════════════════════════════════════════════════════

def stage_esm2_encode(output_dir, config):
    import torch
    out = Path(output_dir)
    embed_path = out / "esm2_embeddings.npy"
    if embed_path.exists():
        embeddings = np.load(embed_path)
        print(f"  ESM-2嵌入已存在: {embeddings.shape}")
        return {"embed_shape": list(embeddings.shape)}

    feat_df = pd.read_csv(out / "feature_matrix.csv")
    import esm
    model_name = config.get("esm2_model", "esm2_t12_35M_UR50D")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if config["device"] == "auto":
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    else:
        device = torch.device(config["device"])
    print(f"  设备: {device}")
    model = model.to(device)

    sequences = feat_df['vh_sequence'].values
    n_seqs = len(sequences)
    repr_layers = [12] if 't12' in model_name else [max(model.num_layers, 1)]
    embed_dim = model.embed_dim
    embeddings = np.zeros((n_seqs, embed_dim), dtype=np.float32)
    batch_size = config.get("esm2_batch_size", 4)

    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        batch_seqs = [(f'seq_{i}', str(sequences[i])) for i in range(start, end)]
        try:
            with torch.no_grad():
                _, _, tokens = batch_converter(batch_seqs)
                tokens = tokens.to(device)
                results = model(tokens, repr_layers=repr_layers)
                repr = results['representations'][repr_layers[0]]
            for i, (_, seq) in enumerate(batch_seqs):
                embeddings[start + i] = repr[i, 1:len(seq)+1].mean(dim=0).cpu().numpy()
        except Exception as e:
            print(f"  批次 {start}-{end} 错误: {e}")
            for i in range(start, end):
                try:
                    with torch.no_grad():
                        _, _, tokens = batch_converter([(f'seq_{i}', str(sequences[i]))])
                        tokens = tokens.to(device)
                        result = model(tokens, repr_layers=repr_layers)
                        embeddings[i] = result['representations'][repr_layers[0]][0, 1:len(str(sequences[i]))+1].mean(dim=0).cpu().numpy()
                except Exception as e2:
                    print(f"  序列 {i} 编码失败: {e2}")
                    embeddings[i] = np.zeros(embed_dim, dtype=np.float32)
        if (start // batch_size) % 50 == 0:
            print(f"  进度: {end}/{n_seqs} ({end/n_seqs*100:.1f}%)")

    np.save(embed_path, embeddings)
    print(f"  ESM-2编码完成: {embeddings.shape}")
    return {"embed_shape": list(embeddings.shape)}

# ═══════════════════════════════════════════════════════════════
# 阶段5: 反事实序列导航
# ═══════════════════════════════════════════════════════════════

def stage_layer3_counterfactual(output_dir, config):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import TSNE
    import lightgbm as lgb
    out = Path(output_dir)
    feat_df = pd.read_csv(out / "feature_matrix.csv")
    embeddings = np.load(out / "esm2_embeddings.npy")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rf2_failed = feat_df[feat_df['rf2_passed'] == False]
    rf2_passed = feat_df[feat_df['rf2_passed'] == True]
    failed_idx = rf2_failed.index.values
    passed_idx = rf2_passed.index.values
    X_all = embeddings.copy()
    Y_binary = feat_df['rf2_passed'].values.astype(int)
    Y_pae = feat_df['rf2_interaction_pae'].values.astype(float)
    print(f"  RF2失败: {len(failed_idx)}, 通过: {len(passed_idx)}")

    def double_ml_cate(X, T, Y, n_folds=5):
        n = len(Y); cate = np.zeros(n); t_stats = np.zeros(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for _, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            T_tr, T_te = T[train_idx], T[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            m_y = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)
            m_y.fit(X_tr, Y_tr); Y_resid = Y_te - m_y.predict(X_te)
            m_t = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)
            m_t.fit(X_tr, T_tr); T_resid = T_te - m_t.predict_proba(X_te)[:, 1]
            ss = np.sum(T_resid**2)
            if ss < 1e-8: continue
            theta = np.sum(T_resid * Y_resid) / ss
            cate[test_idx] = theta
            residuals = Y_resid - theta * T_resid
            se = np.sqrt(np.sum(residuals**2) / (n-1) / ss)
            t_stats[test_idx] = theta / se if se > 0 else 0
        return cate, t_stats

    cate_pae, tstat_pae = double_ml_cate(X_all, Y_binary, Y_pae)
    np.save(out / 'cate_pae.npy', cate_pae)
    np.save(out / 'tstat_pae.npy', tstat_pae)

    pos_cate_results = []
    for pos in range(config.get("max_cdr3_len", 13)):
        mask_pos = feat_df['cdr3_len'] > pos
        if mask_pos.sum() < 100: continue
        pos_data = feat_df[mask_pos]; pos_idx = pos_data.index.values
        for aa in AMINO_ACIDS:
            aa_at_pos = pos_data['cdr3_sequence'].apply(lambda s: s[pos] if pd.notna(s) and len(s) > pos else 'X')
            is_aa = (aa_at_pos == aa).astype(int).values
            if is_aa.sum() < 30: continue
            X_pos = embeddings[pos_idx]; Y_pos = Y_pae[pos_idx]
            X_full = np.hstack([is_aa.reshape(-1, 1), X_pos])
            lr = Ridge(alpha=1.0); lr.fit(X_full, Y_pos)
            cate_val = lr.coef_[0]
            n = len(Y_pos); k = X_full.shape[1]
            residuals = Y_pos - lr.predict(X_full); mse = np.sum(residuals**2) / (n-k-1)
            try:
                inv_diag = np.linalg.inv(X_full.T @ X_full)[0,0]
            except np.linalg.LinAlgError:
                inv_diag = np.linalg.pinv(X_full.T @ X_full)[0,0]
            se = np.sqrt(mse * inv_diag)
            t_stat = cate_val / se if se > 0 else 0
            if abs(t_stat) > 2.0:
                pos_cate_results.append({'position': pos, 'amino_acid': aa, 'CATE': cate_val, 'SE': se, 't_stat': t_stat, 'n_sequences': int(is_aa.sum())})

    pos_cate_df = pd.DataFrame(pos_cate_results).sort_values('CATE') if pos_cate_results else pd.DataFrame()
    if len(pos_cate_df) > 0:
        pos_cate_df.to_csv(out / 'position_specific_cate.csv', index=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        pivot = pos_cate_df.pivot_table(index='amino_acid', columns='position', values='CATE', aggfunc='first')
        sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.1f', ax=ax, cbar_kws={'label': 'CATE'})
        ax.set_title('Position-Specific CATE'); ax.set_xlabel('Position'); ax.set_ylabel('AA')
        plt.tight_layout(); plt.savefig(out / 'layer3_position_cate_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()

    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(embeddings[passed_idx])
    distances, nn_indices = nn.kneighbors(embeddings[failed_idx])
    np.save(out / 'nn_distances.npy', distances)
    np.save(out / 'nn_indices.npy', nn_indices)

    n_process = min(config.get("counterfactual_top_n", 2000), len(failed_idx))
    cf_suggestions = []
    for i in range(n_process):
        row = feat_df.iloc[failed_idx[i]]
        cdr3 = row['cdr3_sequence']
        if pd.isna(cdr3) or len(cdr3) == 0: continue
        suggestions = []
        for pos in range(min(len(cdr3), 13)):
            for aa in AMINO_ACIDS:
                if aa == cdr3[pos]: continue
                mutated = cdr3[:pos] + aa + cdr3[pos+1:]
                pae_change = 0.0
                if len(pos_cate_df) > 0:
                    match = pos_cate_df[(pos_cate_df['position']==pos) & (pos_cate_df['amino_acid']==aa)]
                    if len(match) > 0: pae_change = match.iloc[0]['CATE']
                suggestions.append({'pos': pos, 'orig': cdr3[pos], 'mut': aa, 'cdr3': mutated, 'pae': pae_change})
        suggestions.sort(key=lambda x: x['pae'])
        for rank, s in enumerate(suggestions[:3]):
            nn_idx = nn_indices[i, 0]
            succ_cdr3 = str(feat_df.iloc[passed_idx[nn_idx]]['cdr3_sequence'])
            cf_suggestions.append({
                'sequence_id': int(row['global_sequence_index']), 'original_cdr3': cdr3,
                'rank': rank+1, 'edit': f"Pos{s['pos']} {s['orig']}->{s['mut']}",
                'mutated_cdr3': s['cdr3'], 'predicted_pae_change': round(s['pae'], 2),
                'edit_distance_to_template': sum(a!=b for a,b in zip(s['cdr3'], succ_cdr3)) if len(s['cdr3'])==len(succ_cdr3) else -1,
                'nearest_success_cdr3': succ_cdr3,
            })
    cf_df = pd.DataFrame(cf_suggestions)
    cf_df.to_csv(out / 'counterfactual_suggestions.csv', index=False)

    trunc_results = []
    for i in range(min(n_process, len(failed_idx))):
        row = feat_df.iloc[failed_idx[i]]
        cdr3 = row['cdr3_sequence']
        if pd.isna(cdr3) or len(cdr3) <= 7: continue
        for tl in [6, 7]:
            if len(cdr3) <= tl: continue
            trunc_results.append({'sequence_id': int(row['global_sequence_index']), 'original_cdr3': cdr3, 'original_len': len(cdr3), 'truncated_cdr3': cdr3[:tl], 'target_len': tl})
    pd.DataFrame(trunc_results).to_csv(out / 'truncation_suggestions.csv', index=False)

    sample_size = min(config.get("tsne_sample_size", 3000), len(embeddings))
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeds_2d = tsne.fit_transform(embeddings[sample_idx])
    sample_labels = Y_binary[sample_idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(embeds_2d[sample_labels==0,0], embeds_2d[sample_labels==0,1], c='#e74c3c', alpha=0.3, s=10, label='RF2 Failed')
    ax.scatter(embeds_2d[sample_labels==1,0], embeds_2d[sample_labels==1,1], c='#2ecc71', alpha=0.6, s=20, label='RF2 Passed')
    ax.set_title('ESM-2 t-SNE'); ax.legend()
    plt.tight_layout(); plt.savefig(out / 'layer3_tsne_embedding.png', dpi=150, bbox_inches='tight'); plt.close()

    return {"n_counterfactual_suggestions": len(cf_df), "n_position_cate": len(pos_cate_df)}

# ═══════════════════════════════════════════════════════════════
# 阶段6: 规则合成与输出
# ═══════════════════════════════════════════════════════════════

def stage_layer5_synthesis(output_dir, config):
    out = Path(output_dir)
    feat_df = pd.read_csv(out / "feature_matrix.csv")

    len_pass = feat_df.groupby('cdr3_len').agg(total=('rf2_passed', 'count'), rf2_pass=('rf2_passed', 'sum'), final_cand=('final_candidate', 'sum')).reset_index()
    len_pass['pass_rate'] = len_pass['rf2_pass'] / len_pass['total']
    valid_lengths = len_pass[len_pass['pass_rate'] > 0.10]['cdr3_len'].tolist()

    first_res = feat_df.groupby('first_residue').agg(total=('rf2_passed', 'count'), rf2_pass=('rf2_passed', 'sum')).reset_index()
    first_res['rate'] = first_res['rf2_pass'] / first_res['total']
    first_whitelist = first_res[first_res['rate'] > 0.40]['first_residue'].tolist()

    last_res = feat_df.groupby('last_residue').agg(total=('rf2_passed', 'count'), rf2_pass=('rf2_passed', 'sum')).reset_index()
    last_res['rate'] = last_res['rf2_pass'] / last_res['total']
    last_whitelist = last_res[last_res['rate'] > 0.10]['last_residue'].tolist()

    best_aromatic = 0.20
    for t in [0.15, 0.20, 0.25]:
        if feat_df.loc[feat_df['aromatic_ratio'] >= t, 'rf2_passed'].mean() > 0.15: best_aromatic = t; break
    best_glycine = 0.20
    for t in [0.20, 0.15, 0.12]:
        if feat_df.loc[feat_df['glycine_ratio'] <= t, 'rf2_passed'].mean() > 0.12: best_glycine = t; break
    best_serine = 0.15
    for t in [0.20, 0.15, 0.10]:
        if feat_df.loc[feat_df['serine_ratio'] <= t, 'rf2_passed'].mean() > 0.15: best_serine = t; break

    anti_patterns = []
    for pattern, col in [('GGG', 'has_ggg'), ('SSS', 'has_sss'), ('LL', 'has_ll')]:
        if col in feat_df.columns and feat_df[col].sum() > 0:
            with_rate = feat_df.loc[feat_df[col], 'rf2_passed'].mean()
            without_rate = feat_df.loc[~feat_df[col], 'rf2_passed'].mean()
            if with_rate < without_rate * 0.5: anti_patterns.append(pattern)

    success_df = feat_df[feat_df['rf2_passed'] == True]
    success_templates = {}
    for length in sorted(set(valid_lengths)):
        len_s = success_df[success_df['cdr3_len'] == length]
        if len(len_s) == 0: continue
        success_templates[f'length_{length}'] = len_s['cdr3_sequence'].value_counts().head(10).index.tolist()

    salvage_edits = []
    cf_path = out / 'counterfactual_suggestions.csv'
    if cf_path.exists():
        cf_df = pd.read_csv(cf_path)
        for pattern, count in cf_df['edit'].value_counts().head(30).items():
            mean_pae = cf_df[cf_df['edit'] == pattern]['predicted_pae_change'].mean()
            if mean_pae < -0.5:
                salvage_edits.append({'pattern': pattern, 'count': int(count), 'mean_pae_change': round(mean_pae, 2)})

    trunc_to_7_count = 0
    trunc_path = out / 'truncation_suggestions.csv'
    if trunc_path.exists():
        trunc_df = pd.read_csv(trunc_path)
        trunc_to_7_count = len(trunc_df[(trunc_df['original_len'] >= 10) & (trunc_df['target_len'] == 7)])

    cox_hr = {}
    cox_path = out / 'cox_hazard_ratios.csv'
    if cox_path.exists():
        cox_df = pd.read_csv(cox_path)
        for _, r in cox_df.iterrows(): cox_hr[r['variable']] = round(r['HR'], 2)

    ate_vals = {}
    ate_path = out / 'ate_estimates.csv'
    if ate_path.exists():
        ate_df = pd.read_csv(ate_path)
        for _, r in ate_df.iterrows():
            if r['outcome'] == 'rf2_interaction_pae' and r['p_value'] < 0.05:
                ate_vals[r['treatment']] = round(r['ATE'], 2)

    design_strategy = {
        'strategy_name': 'CSC-O_v1',
        'description': 'Causal-Stratified Counterfactual Optimization for Antibody CDR3 Design',
        'version': '1.0',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'data_source': str(config.get('input_file', '')),
        'data_size': len(feat_df),
        'hard_constraints': {
            'cdr3_length_allowed': [int(l) for l in valid_lengths],
            'cdr3_length_preferred': [l for l in [6,7] if l in valid_lengths],
            'cdr3_first_residue_whitelist': first_whitelist,
            'cdr3_last_residue_whitelist': last_whitelist,
            'cdr3_min_positive_count': 0,
            'cdr3_min_aromatic_first': True,
        },
        'soft_preferences': {
            'aromatic_min_ratio': best_aromatic,
            'glycine_max_ratio': best_glycine,
            'serine_max_ratio': best_serine,
            'proline_max_count': 1,
        },
        'anti_patterns': anti_patterns,
        'success_templates': success_templates,
        'salvage_edits': salvage_edits[:20],
        'truncation_rule': {
            'if_cdr3_length_ge_10': 'truncate_to_7',
            'evidence_count': trunc_to_7_count,
        },
        'cox_hazard_ratios': cox_hr,
        'ate_estimates': ate_vals,
    }

    with open(out / 'design_strategy.json', 'w', encoding='utf-8') as f:
        json.dump(design_strategy, f, indent=2, ensure_ascii=False)

    lines = ['# CSC-O Design Strategy', f'# Generated: {datetime.now().strftime("%Y-%m-%d")}', '']
    lines.append(f'CDR3_LENGTH_ALLOWED = {[int(l) for l in valid_lengths]}')
    lines.append(f'CDR3_FIRST_RESIDUE_WHITELIST = {first_whitelist}')
    lines.append(f'CDR3_LAST_RESIDUE_WHITELIST = {last_whitelist}')
    lines.append(f'AROMATIC_MIN_RATIO = {best_aromatic}')
    lines.append(f'GLYCINE_MAX_RATIO = {best_glycine}')
    lines.append(f'SERINE_MAX_RATIO = {best_serine}')
    lines.append(f'PROLINE_MAX_COUNT = 1')
    for ap in anti_patterns: lines.append(f'FORBIDDEN_PATTERN = {ap}')
    for lk, tmpls in success_templates.items():
        for t in tmpls: lines.append(f'TEMPLATE_{lk.upper()} = {t}')
    for se in salvage_edits[:15]: lines.append(f'SALVAGE_EDIT = {se["pattern"]}  # count={se["count"]}, pae={se["mean_pae_change"]}')
    lines.append('TRUNCATION_RULE = IF_LEN_GE_10_THEN_TRUNCATE_TO_7')
    with open(out / 'design_strategy.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    report = ['='*80, 'CSC-O 综合分析报告', '='*80, '']
    report.append(f'数据来源: {config.get("input_file", "")}')
    report.append(f'数据量: {len(feat_df)} 条')
    report.append(f'RF2通过率: {feat_df["rf2_passed"].mean()*100:.1f}%')
    report.append(f'最终候选率: {feat_df["final_candidate"].mean()*100:.2f}%')
    report.append(f'CDR3长度允许值: {valid_lengths}')
    report.append(f'首残基白名单: {first_whitelist}')
    report.append(f'尾残基白名单: {last_whitelist}')
    report.append(f'反模式: {anti_patterns}')
    report.append(f'高频挽救编辑: {[se["pattern"] for se in salvage_edits[:5]]}')
    report.append('='*80)
    with open(out / 'csco_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"  策略文件: design_strategy.json, design_strategy.txt, csco_analysis_report.txt")
    return {"valid_lengths": valid_lengths, "anti_patterns": anti_patterns, "n_salvage_edits": len(salvage_edits)}

# ═══════════════════════════════════════════════════════════════
# 主管线
# ═══════════════════════════════════════════════════════════════

STAGE_MAP = {
    "data_engineering":       stage_data_engineering,
    "layer1_stratified":      stage_layer1_stratified,
    "layer2_causal":          stage_layer2_causal,
    "layer3_esm2_encode":     stage_esm2_encode,
    "layer3_counterfactual":  stage_layer3_counterfactual,
    "layer5_synthesis":       stage_layer5_synthesis,
}

def run_pipeline(config):
    tracker = ProgressTracker(config["work_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = None
    for stage_name in STAGE_NAMES:
        if config["stage_resume"] and tracker.is_completed(stage_name):
            print(f"  跳过已完成阶段: {stage_name}")
            continue

        t0 = tracker.start_stage(stage_name)
        try:
            if stage_name == "data_engineering":
                adapter = DataAdapter(config["input_file"])
                df = adapter.validate()
                metrics = stage_data_engineering(df, output_dir, config)
            elif stage_name == "layer1_stratified":
                metrics = stage_layer1_stratified(output_dir, config)
            elif stage_name == "layer2_causal":
                metrics = stage_layer2_causal(output_dir, config)
            elif stage_name == "layer3_esm2_encode":
                metrics = stage_esm2_encode(output_dir, config)
            elif stage_name == "layer3_counterfactual":
                metrics = stage_layer3_counterfactual(output_dir, config)
            elif stage_name == "layer5_synthesis":
                metrics = stage_layer5_synthesis(output_dir, config)
            else:
                metrics = {}
            tracker.complete_stage(stage_name, t0, metrics)
        except Exception as e:
            tracker.log_error(stage_name, e)
            print(f"  错误: {e}")
            traceback.print_exc()
            raise

    print(f"\n{'='*60}")
    print(f"管线完成！输出目录: {output_dir.absolute()}")
    print(f"完成阶段: {[s['name'] for s in tracker.state['completed_stages']]}")
    print(f"{'='*60}")
    return output_dir

# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSC-O Antibody Design Pipeline")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径 (CSV/XLSX/XLS/TSV)")
    parser.add_argument("--output", "-o", default="./output", help="输出目录")
    parser.add_argument("--work", "-w", default="./work", help="工作目录")
    parser.add_argument("--resume", "-r", action="store_true", help="断点续跑")
    parser.add_argument("--device", "-d", default="auto", choices=["auto","cuda","mps","cpu"])
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="ESM-2批大小")
    parser.add_argument("--top-n", "-n", type=int, default=2000, help="反事实编辑处理序列数")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["input_file"] = args.input
    config["output_dir"] = args.output
    config["work_dir"] = args.work
    config["stage_resume"] = args.resume
    config["device"] = args.device
    config["esm2_batch_size"] = args.batch_size
    config["counterfactual_top_n"] = args.top_n

    run_pipeline(config)
