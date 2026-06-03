#!/usr/bin/env python3
"""
CSC-O Pipeline — Causal-Stratified Counterfactual Optimization
统一入口：自动检测输入格式、补全缺失列、顺序执行全部6个阶段

重构版本 (rf3-v2):
- 优化目标从 rf2_passed 切换为 final_candidate（可配置）
- 统一使用 csco_config 公共模块，消除代码克隆
- 修复因果推断 Treatment=Outcome Bug
- 增强 R-learner 数值稳定性
- 模块化拆分，提升可维护性
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

from csco_config import (
    AMINO_ACIDS, AROMATIC, POSITIVE, NEGATIVE, HYDROPHOBIC, GLYCINE, SERINE, PROLINE,
    ESM2_MODEL_REGISTRY, DEFAULT_CONFIG, STAGE_NAMES,
    TREATMENT_VARS, TREATMENT_VARS_EXTENDED, CONFOUNDER_COLS, CDR3_FEATURE_COLS,
    COLUMN_SCHEMA, COLUMN_ALIASES,
    extract_cdr3_features, get_optimization_target,
)

# ═══════════════════════════════════════════════════════════════
# 进度持久化
# ═══════════════════════════════════════════════════════════════

class ProgressTracker:
    def __init__(self, work_dir):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.work_dir / "pipeline_state.json"
        self.notify_dir = self.work_dir / "notifications"
        self.notify_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"started_at": None, "completed_stages": [], "current_stage": None, "errors": []}

    def save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def _notify(self, event_type, stage, detail=""):
        notify_file = self.notify_dir / f"{event_type}_{stage}_{int(time.time())}.txt"
        lines = [
            f"EVENT: {event_type}",
            f"STAGE: {stage}",
            f"TIME: {datetime.now().isoformat()}",
            f"DETAIL: {detail}",
            f"COMPLETED: {[s['name'] for s in self.state.get('completed_stages', [])]}",
        ]
        with open(notify_file, 'w') as f:
            f.write('\n'.join(lines))
        latest_file = self.notify_dir / "LATEST_STATUS.txt"
        with open(latest_file, 'w') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {event_type}: {stage}\n")

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
        detail = f"耗时 {elapsed:.1f}s"
        if metrics:
            detail += f" | metrics={metrics}"
        self._notify("COMPLETE", name, detail)
        print(f"<<< 阶段完成: {name} | 耗时: {elapsed:.1f}s\n")

    def log_error(self, stage, error):
        self.state["errors"].append({
            "stage": stage,
            "time": datetime.now().isoformat(),
            "error": str(error),
            "traceback": traceback.format_exc()
        })
        self.save()
        self._notify("ERROR", stage, str(error)[:200])

# ═══════════════════════════════════════════════════════════════
# 数据适配层
# ═══════════════════════════════════════════════════════════════

class DataAdapter:
    def __init__(self, input_path, config=None):
        self.input_path = Path(input_path)
        self.config = config or {}
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

        # 始终基于原始指标重新计算rf2_passed_filter（使用可配置阈值）
        if 'rf2_pred_lddt' in self.df.columns and 'rf2_interaction_pae' in self.df.columns:
            rf2_lddt_thresh = self.config.get('rf2_lddt_threshold', 0.86)
            rf2_pae_thresh = self.config.get('rf2_pae_threshold', 10.0)
            rf2_rmsd_thresh = self.config.get('rf2_rmsd_threshold', 2.5)
            lddt_ok = self.df['rf2_pred_lddt'] >= rf2_lddt_thresh
            pae_ok = self.df['rf2_interaction_pae'] <= rf2_pae_thresh
            rmsd_ok = self.df['rf2_framework_aligned_cdr_rmsd'].fillna(0) <= rf2_rmsd_thresh
            old_pass = self.df['rf2_passed_filter'].sum() if 'rf2_passed_filter' in self.df.columns else 0
            self.df['rf2_passed_filter'] = lddt_ok & pae_ok & rmsd_ok
            n_pass = self.df['rf2_passed_filter'].sum()
            print(f"  RF2筛选(lddt≥{rf2_lddt_thresh}, pae≤{rf2_pae_thresh}, rmsd≤{rf2_rmsd_thresh}): {n_pass}/{len(self.df)} 通过 (原{old_pass}条)")

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

    survival_records = _build_survival_data(feat_df)
    surv_df = pd.DataFrame(survival_records)
    surv_df.to_csv(out / "survival_data.csv", index=False)

    print(f"  Feature matrix: {feat_df.shape}, Survival data: {surv_df.shape}")
    return {"n_samples": len(df), "n_features": len(feat_df.columns)}

def _build_survival_data(feat_df):
    records = []
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
        records.append({
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
    return records

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

    _plot_attrition_funnel(out, stages=['RF2', 'AF3', 'Schrödinger', 'Desmond', 'Candidate'],
                           counts=[s1.sum(), s2.sum(), s3.sum(), s4.sum(), sv.sum()], total=total)

    cox_results = _run_cox_regression(surv_df, out)
    _plot_km_curves(surv_df, out)
    _run_threshold_sensitivity(feat_df, out)

    target_label = config.get('optimization_target', 'final_candidate')
    _plot_cdr3_length_analysis(feat_df, out, target_label=target_label)

    return {"cox_concordance": cox_results['concordance']}

def _plot_attrition_funnel(out, stages, counts, total):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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

def _run_cox_regression(surv_df, out):
    from lifelines import CoxPHFitter
    cox_vars_centered = ['cdr3_len', 'positive_ratio', 'aromatic_ratio', 'glycine_ratio', 'serine_ratio', 'hydrophobic_ratio']
    cox_df = surv_df.copy()
    for v in cox_vars_centered:
        cox_df[f'{v}_centered'] = cox_df[v] - cox_df[v].mean()
    cox_var_names = [f'{v}_centered' for v in cox_vars_centered]
    cox_data = cox_df[['time', 'event'] + cox_var_names].dropna()
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='time', event_col='event')
    cph.print_summary()

    hr_results = []
    for var in cox_var_names:
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

    return {'concordance': cph.concordance_index_}

def _plot_km_curves(surv_df, out):
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    for label, mask in {'5-7': surv_df['cdr3_len'].isin([5,6,7]), '8-9': surv_df['cdr3_len'].isin([8,9]), '10+': surv_df['cdr3_len']>=10}.items():
        kmf.fit(surv_df.loc[mask, 'time'], surv_df.loc[mask, 'event'], label=label)
        kmf.plot_survival_function(ax=ax)
    ax.set_title('KM Survival by CDR3 Length')
    plt.tight_layout()
    plt.savefig(out / 'layer1_km_by_cdr3_length.png', dpi=150, bbox_inches='tight')
    plt.close()

def _run_threshold_sensitivity(feat_df, out):
    lddt_values = feat_df['rf2_pred_lddt'].values.astype(float)
    pae_values = feat_df['rf2_interaction_pae'].values.astype(float)
    final_candidates = feat_df['final_candidate'].values.astype(bool)
    sens_results = []
    for thresh in np.arange(0.82, 0.92, 0.005):
        pred_pass = (lddt_values >= thresh) & (pae_values <= 10.0)
        tp = np.sum(pred_pass & final_candidates)
        fp = np.sum(pred_pass & ~final_candidates)
        fn = np.sum(~pred_pass & final_candidates)
        sens_results.append({
            'threshold': thresh, 'pass_rate': pred_pass.sum() / len(pred_pass),
            'sensitivity': tp / max(tp + fn, 1), 'precision': tp / max(tp + fp, 1),
        })
    pd.DataFrame(sens_results).to_csv(out / 'threshold_sensitivity.csv', index=False)

def _plot_cdr3_length_analysis(feat_df, out, target_label='final_candidate'):
    target_col = 'final_candidate' if target_label == 'final_candidate' and 'final_candidate' in feat_df.columns else 'rf2_passed'
    len_pass = feat_df.groupby('cdr3_len').agg(
        total=(target_col, 'count'), rf2_pass=('rf2_passed', 'sum'), final_cand=('final_candidate', 'sum')
    ).reset_index()
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

    outcome_binary = 'rf2_passed'
    outcome_continuous = 'rf2_interaction_pae'

    feat_df['first_is_aromatic'] = feat_df['first_residue'].isin(['Y', 'W', 'F']).astype(int)
    feat_df['last_is_YH'] = feat_df['last_residue'].isin(['Y', 'H']).astype(int)

    causal_cols = TREATMENT_VARS_EXTENDED + [outcome_binary, outcome_continuous, 'backbone_id']
    causal_data = feat_df[causal_cols].copy()
    causal_data[outcome_binary] = causal_data[outcome_binary].astype(int)

    domain_constraints = [(o, t) for o in [outcome_binary, outcome_continuous] for t in TREATMENT_VARS_EXTENDED]
    dag, nodes, sep_sets = _pc_algorithm(causal_data, alpha=0.01, domain_constraints=domain_constraints)
    _plot_causal_dag(dag, nodes, out, TREATMENT_VARS_EXTENDED, outcome_binary, outcome_continuous)

    ate_results, stratified_results = _estimate_all_ate(feat_df, out, outcome_binary, outcome_continuous)
    _plot_ate_forest(ate_results, out, outcome_binary)

    return {"n_ate_results": len(ate_results), "n_stratified_results": len(stratified_results)}

def _pc_algorithm(data, alpha=0.01, domain_constraints=None):
    from scipy import stats
    from itertools import combinations
    from sklearn.linear_model import LinearRegression

    def partial_corr_test(x, y, z_data, alpha=0.05):
        if z_data.shape[1] == 0:
            return stats.pearsonr(x, y)
        lr = LinearRegression()
        lr.fit(z_data, x); res_x = x - lr.predict(z_data)
        lr.fit(z_data, y); res_y = y - lr.predict(z_data)
        return stats.pearsonr(res_x, res_y)

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

def _plot_causal_dag(dag, nodes, out, treatment_vars_ext, outcome_binary, outcome_continuous):
    pos = {}
    for i, node in enumerate(TREATMENT_VARS[:6]): pos[node] = (i*2, 2)
    for i, node in enumerate(['first_is_aromatic', 'last_is_YH']): pos[node] = (i*2+3, 1)
    for i, node in enumerate([outcome_binary, outcome_continuous]): pos[node] = (i*2+4, 0)
    pos['backbone_id'] = (0, 0.5)

    fig, ax = plt.subplots(figsize=(14, 10))
    for node_name, (x, y) in pos.items():
        if node_name in nodes:
            color = '#e74c3c' if node_name in [outcome_binary, outcome_continuous] else '#3498db' if node_name in treatment_vars_ext else '#95a5a6'
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

def _estimate_all_ate(feat_df, out, outcome_binary, outcome_continuous):
    from scipy import stats
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_predict

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
    for tv in TREATMENT_VARS_EXTENDED:
        confounders = ['backbone_id']
        if tv != 'cdr3_len':
            confounders.append('cdr3_len')
        if tv in ['cdr3_len', 'proline_count']:
            feat_df[f'{tv}_binary'] = (feat_df[tv] > feat_df[tv].median()).astype(int)
            ate, _ = backdoor_ate(feat_df, f'{tv}_binary', outcome_binary, confounders, binary_treatment=True)
            ate_results.append({'treatment': tv, 'outcome': outcome_binary, 'ATE': ate, 'SE': 0, 't_stat': 0, 'p_value': 0, 'CI_lower': 0, 'CI_upper': 0})
        ate_cont, info = backdoor_ate(feat_df, tv, outcome_continuous, confounders)
        se, t_stat, p_val, ci_l, ci_u = info if info else (0,0,0,0,0)
        ate_results.append({'treatment': tv, 'outcome': outcome_continuous, 'ATE': ate_cont, 'SE': se, 't_stat': t_stat, 'p_value': p_val, 'CI_lower': ci_l, 'CI_upper': ci_u})
    ate_df = pd.DataFrame(ate_results)
    ate_df.to_csv(out / 'ate_estimates.csv', index=False)

    stratified_results = _estimate_stratified_ate(feat_df, out, outcome_continuous)
    return ate_results, stratified_results


def _estimate_stratified_ate(feat_df, out, outcome_continuous):
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    target_lengths = sorted(feat_df['cdr3_len'].unique())
    stratified_results = []

    for length in target_lengths:
        sub = feat_df[feat_df['cdr3_len'] == length]
        if len(sub) < 50:
            continue

        for tv in TREATMENT_VARS_EXTENDED:
            if tv == 'cdr3_len':
                continue
            confounders = ['backbone_id']
            try:
                T = sub[tv].values.reshape(-1, 1).astype(float)
                Y = sub[outcome_continuous].values.astype(float)
                X_c = sub[confounders].values.astype(float)
                valid = ~(np.isnan(T.flatten()) | np.isnan(Y) | np.any(np.isnan(X_c), axis=1))
                T = T[valid]; Y = Y[valid]; X_c = X_c[valid]
                if len(Y) < 30:
                    continue
                X_full = np.hstack([T, X_c])
                lr = LinearRegression().fit(X_full, Y)
                ate = lr.coef_[0]
                n = len(Y); k = X_full.shape[1]
                res = Y - lr.predict(X_full)
                mse = np.sum(res**2) / max(n - k - 1, 1)
                try:
                    inv_d = np.linalg.inv(X_full.T @ X_full)[0, 0]
                except np.linalg.LinAlgError:
                    inv_d = np.linalg.pinv(X_full.T @ X_full)[0, 0]
                se = np.sqrt(mse * inv_d)
                t_stat = ate / se
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k - 1))
                stratified_results.append({
                    'cdr3_len': int(length), 'n_samples': len(Y),
                    'treatment': tv, 'outcome': outcome_continuous,
                    'ATE': round(ate, 4), 'SE': round(se, 4),
                    't_stat': round(t_stat, 2), 'p_value': p_val,
                })
            except Exception:
                continue

    if stratified_results:
        strat_df = pd.DataFrame(stratified_results)
        strat_df.to_csv(out / 'stratified_ate_estimates.csv', index=False)
    return stratified_results

def _plot_ate_forest(ate_results, out, outcome_binary):
    ate_df = pd.DataFrame(ate_results)
    fig, ax = plt.subplots(figsize=(12, 8))
    binary_ate = ate_df[ate_df['outcome'] == outcome_binary].sort_values('ATE', ascending=True)
    ax.barh(range(len(binary_ate)), binary_ate['ATE'], color=['#e74c3c' if p<0.05 else '#95a5a6' for p in binary_ate['p_value']], alpha=0.7, height=0.6)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_yticks(range(len(binary_ate))); ax.set_yticklabels(binary_ate['treatment'])
    ax.set_xlabel('ATE on RF2 Pass Rate'); ax.set_title('Causal ATE Estimates')
    plt.tight_layout()
    plt.savefig(out / 'layer2_ate_forest.png', dpi=150, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════
# 阶段4: ESM-2编码
# ═══════════════════════════════════════════════════════════════

def _encode_single_model(model_name, sequences, device, batch_size):
    import torch, esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    n_seqs = len(sequences)
    info = ESM2_MODEL_REGISTRY.get(model_name, {"layers": model.num_layers, "dim": model.embed_dim})
    repr_layers = [info["layers"]]
    embed_dim = info["dim"]
    embeddings = np.zeros((n_seqs, embed_dim), dtype=np.float32)
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
                except Exception:
                    embeddings[i] = np.zeros(embed_dim, dtype=np.float32)
        if (start // batch_size) % 50 == 0:
            print(f"    [{model_name}] 进度: {end}/{n_seqs} ({end/n_seqs*100:.1f}%)")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings

def _fuse_embeddings(embed_dict, strategy, target_dim=None):
    parts = list(embed_dict.values())
    if strategy == "concat":
        fused = np.hstack(parts)
        print(f"  融合策略=concat, 输出维度: {fused.shape[1]}")
    elif strategy == "average":
        max_dim = max(p.shape[1] for p in parts)
        padded = []
        for p in parts:
            if p.shape[1] < max_dim:
                pad = np.zeros((p.shape[0], max_dim - p.shape[1]), dtype=np.float32)
                padded.append(np.hstack([p, pad]))
            else:
                padded.append(p)
        fused = np.mean(np.stack(padded, axis=0), axis=0)
        print(f"  融合策略=average, 输出维度: {fused.shape[1]}")
    elif strategy == "pca_concat":
        from sklearn.decomposition import PCA
        reduced = []
        for name, emb in embed_dict.items():
            n_components = min(target_dim or 256, emb.shape[1], emb.shape[0])
            pca = PCA(n_components=n_components, random_state=42)
            reduced.append(pca.fit_transform(emb))
            print(f"    PCA: {name} {emb.shape[1]}→{n_components} (explained_var={pca.explained_variance_ratio_.sum():.3f})")
        fused = np.hstack(reduced)
        print(f"  融合策略=pca_concat, 输出维度: {fused.shape[1]}")
    else:
        raise ValueError(f"未知融合策略: {strategy}")
    return fused

def stage_esm2_encode(output_dir, config):
    import torch
    out = Path(output_dir)
    embed_path = out / "esm2_embeddings.npy"
    model_names = config.get("esm2_models", [config.get("esm2_model", "esm2_t12_35M_UR50D")])
    fusion_strategy = config.get("esm2_fusion", "concat")
    if embed_path.exists():
        embeddings = np.load(embed_path)
        print(f"  ESM-2嵌入已存在: {embeddings.shape}")
        return {"embed_shape": list(embeddings.shape), "models_used": model_names, "fusion": fusion_strategy}

    feat_df = pd.read_csv(out / "feature_matrix.csv")
    if config["device"] == "auto":
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    else:
        device = torch.device(config["device"])
    print(f"  设备: {device}")

    sequences = feat_df['vh_sequence'].values
    batch_size = config.get("esm2_batch_size", 4)
    embed_dict = {}
    for model_name in model_names:
        print(f"  编码模型: {model_name}")
        t0 = time.time()
        emb = _encode_single_model(model_name, sequences, device, batch_size)
        elapsed = time.time() - t0
        print(f"    完成: shape={emb.shape}, 耗时={elapsed:.1f}s")
        np.save(out / f"esm2_{model_name}.npy", emb)
        embed_dict[model_name] = emb

    if len(embed_dict) == 1:
        embeddings = list(embed_dict.values())[0]
    else:
        pca_dim = config.get("esm2_pca_dim", 256)
        embeddings = _fuse_embeddings(embed_dict, fusion_strategy, target_dim=pca_dim)

    np.save(embed_path, embeddings)
    print(f"  ESM-2编码完成: {embeddings.shape} (models={model_names}, fusion={fusion_strategy})")
    return {"embed_shape": list(embeddings.shape), "models_used": model_names, "fusion": fusion_strategy}

# ═══════════════════════════════════════════════════════════════
# 阶段5: 反事实序列导航
# ═══════════════════════════════════════════════════════════════

def _double_ml_cate(X, T, Y, n_folds=5, method="causal_forest", inner_n_jobs=1):
    n = len(Y); cate = np.zeros(n); t_stats = np.zeros(n); se_arr = np.zeros(n)
    kf = __import__('sklearn.model_selection', fromlist=['KFold']).KFold(n_splits=n_folds, shuffle=True, random_state=42)
    T_arr = np.asarray(T).ravel()
    is_discrete_t = len(np.unique(T_arr)) <= 10

    if method == "causal_forest":
        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            if is_discrete_t:
                model_t = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            else:
                model_t = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            cf = CausalForestDML(
                model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
                model_t=model_t,
                discrete_treatment=is_discrete_t,
                n_estimators=200, max_depth=4, min_samples_leaf=20,
                random_state=42, cv=3, n_jobs=inner_n_jobs,
            )
            cf.fit(Y, T, X=X, W=None)
            cate = cf.effect(X).flatten()
            se_arr = np.sqrt(np.array(cf.effect_inference(X).var.tolist()))
            t_stats = cate / np.where(se_arr > 1e-8, se_arr, 1e-8)
            print(f"    CausalForestDML: CATE range [{cate.min():.3f}, {cate.max():.3f}], mean={cate.mean():.3f}")
            return cate, t_stats, se_arr
        except (ImportError, Exception) as e:
            print(f"    CausalForestDML失败: {e}, 降级到R-learner")
            method = "r_learner"

    if method == "r_learner":
        import lightgbm as lgb
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            T_tr, T_te = T[train_idx], T[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            m_y = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbose=-1, n_jobs=inner_n_jobs, random_state=42)
            m_y.fit(X_tr, Y_tr); Y_resid = Y_te - m_y.predict(X_te)
            m_t = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, verbose=-1, n_jobs=inner_n_jobs, random_state=42)
            m_t.fit(X_tr, T_tr)
            ps = m_t.predict_proba(X_te)[:, 1]
            ps = np.clip(ps, 0.1, 0.9)
            T_resid = T_te - ps
            ss = np.sum(T_resid**2)
            if ss < 1e-8: continue
            cate_fold = Y_resid * T_resid / (T_resid**2 + 1e-6)
            cate_fold = np.clip(cate_fold, -50, 50)
            # 逐折日志：Y残差、T残差、倾向得分、Robinson变换统计
            print(f"      fold {fold_i+1}/{n_folds}: Y_resid range=[{Y_resid.min():.2f},{Y_resid.max():.2f}] mean={Y_resid.mean():.3f}, "
                  f"T_resid range=[{T_resid.min():.2f},{T_resid.max():.2f}] mean={T_resid.mean():.3f}, "
                  f"ps range=[{ps.min():.3f},{ps.max():.3f}] mean={ps.mean():.3f}, "
                  f"Robinson_cate range=[{cate_fold.min():.2f},{cate_fold.max():.2f}] mean={cate_fold.mean():.3f}")
            hetero_model = GBR(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            hetero_model.fit(X_te, cate_fold)
            cate[test_idx] = hetero_model.predict(X_te)
            cate[test_idx] = np.clip(cate[test_idx], -50, 50)
            residuals = Y_resid - cate[test_idx] * T_resid
            fold_se = np.sqrt(np.sum(residuals**2) / (n-1) / max(ss, 1e-8))
            se_arr[test_idx] = fold_se
            t_stats[test_idx] = cate[test_idx] / fold_se if fold_se > 0 else 0
            n_sig = int(np.sum(np.abs(t_stats[test_idx]) > 1.96))
            print(f"      fold {fold_i+1}: hetero_CATE range=[{cate[test_idx].min():.2f},{cate[test_idx].max():.2f}] mean={cate[test_idx].mean():.3f}, "
                  f"SE={fold_se:.4f}, significant={n_sig}/{len(test_idx)} ({n_sig/len(test_idx)*100:.1f}%)")
        print(f"    R-learner: CATE range [{cate.min():.3f}, {cate.max():.3f}], mean={cate.mean():.3f}, "
              f"pct_significant={np.mean(np.abs(t_stats)>1.96)*100:.1f}%")
        return cate, t_stats, se_arr

    import lightgbm as lgb
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        T_tr, T_te = T[train_idx], T[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        m_y = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, verbose=-1, n_jobs=inner_n_jobs, random_state=42)
        m_y.fit(X_tr, Y_tr); Y_resid = Y_te - m_y.predict(X_te)
        m_t = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, verbose=-1, n_jobs=inner_n_jobs, random_state=42)
        m_t.fit(X_tr, T_tr); T_resid = T_te - m_t.predict_proba(X_te)[:, 1]
        ss = np.sum(T_resid**2)
        if ss < 1e-8: continue
        theta = np.sum(T_resid * Y_resid) / ss
        cate[test_idx] = theta
        residuals = Y_resid - theta * T_resid
        se = np.sqrt(np.sum(residuals**2) / (n-1) / ss)
        se_arr[test_idx] = se
        t_stats[test_idx] = theta / se if se > 0 else 0
    print(f"    PLR Double ML: CATE={cate.mean():.3f} (constant)")
    return cate, t_stats, se_arr

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

    Y_binary, Y_pae, target_label = get_optimization_target(feat_df, config)

    target_col = 'final_candidate' if target_label == 'final_candidate' else 'rf2_passed'
    target_failed = feat_df[feat_df[target_col] == False]
    target_passed = feat_df[feat_df[target_col] == True]
    failed_idx = target_failed.index.values
    passed_idx = target_passed.index.values
    X_all = embeddings.copy()
    print(f"  目标={target_label}: 失败={len(failed_idx)}, 通过={len(passed_idx)}")

    cate_method = config.get("cate_method", "causal_forest")
    cate_pae, tstat_pae, se_pae = _double_ml_cate(X_all, Y_binary, Y_pae, method=cate_method, inner_n_jobs=-1)
    np.save(out / 'cate_pae.npy', cate_pae)
    np.save(out / 'tstat_pae.npy', tstat_pae)
    np.save(out / 'se_pae.npy', se_pae)

    _run_multi_treatment_cate(feat_df, X_all, Y_pae, out, cate_method)
    _run_subgroup_discovery(feat_df, cate_pae, out, config)

    pos_cate_df = _run_position_specific_cate(feat_df, embeddings, Y_pae, out, config)

    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(embeddings[passed_idx])
    distances, nn_indices = nn.kneighbors(embeddings[failed_idx])
    np.save(out / 'nn_distances.npy', distances)
    np.save(out / 'nn_indices.npy', nn_indices)

    _generate_counterfactual_suggestions(feat_df, failed_idx, passed_idx, cate_pae, se_pae,
                                          pos_cate_df, nn_indices, out, config)

    _generate_truncation_suggestions(feat_df, failed_idx, out, config)

    _plot_tsne(embeddings, Y_binary, out, config)

    return {"n_counterfactual_suggestions": len(pd.read_csv(out / 'counterfactual_suggestions.csv')) if (out / 'counterfactual_suggestions.csv').exists() else 0,
            "n_position_cate": len(pos_cate_df)}

def _run_multi_treatment_cate(feat_df, X_all, Y_pae, out, cate_method):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    treatment_cols = [c for c in TREATMENT_VARS_EXTENDED if c in feat_df.columns]
    confounder_cols = [c for c in CONFOUNDER_COLS if c in feat_df.columns]

    numeric_conf_cols = [c for c in confounder_cols if c != 'backbone_id']
    conf_numeric = feat_df[numeric_conf_cols].values.astype(np.float32) if numeric_conf_cols else np.empty((len(feat_df), 0), dtype=np.float32)
    backbone_codes = feat_df['backbone_id'].astype('category').cat.codes.values.astype(np.float32).reshape(-1, 1) if 'backbone_id' in confounder_cols else np.empty((len(feat_df), 0), dtype=np.float32)
    conf_data = np.hstack([backbone_codes, conf_numeric]) if conf_numeric.shape[1] > 0 or backbone_codes.shape[1] > 0 else None

    X_all_f = np.ascontiguousarray(X_all, dtype=np.float32)
    Y_pae_f = np.ascontiguousarray(Y_pae, dtype=np.float32)

    multi_treatment_results = []
    total = len(treatment_cols)
    print(f"  === Multi-Treatment CATE 开始: {total}个treatment变量, 方法={cate_method} ===")
    print(f"  混杂变量: {confounder_cols}")
    print(f"  特征维度: X_all={X_all_f.shape[1]}, conf_data={conf_data.shape[1] if conf_data is not None else 0}")
    for i, t_col in enumerate(treatment_cols):
        t0 = time.time()
        try:
            T_var = feat_df[t_col].values.astype(np.float64)
            if T_var.std() < 1e-8:
                print(f"  [{i+1}/{total}] {t_col}: 跳过(方差≈0)")
                continue
            T_binary = (T_var > np.median(T_var)).astype(int) if len(np.unique(T_var)) > 5 else T_var.astype(int)
            n_treat = int(T_binary.sum())
            n_control = len(T_binary) - n_treat
            treat_rate = n_treat / len(T_binary) * 100
            print(f"  [{i+1}/{total}] {t_col}: treatment={n_treat}({treat_rate:.1f}%), control={n_control}, T_var range=[{T_var.min():.3f},{T_var.max():.3f}]")
            X_conf = np.hstack([X_all_f, conf_data]) if conf_data is not None else X_all_f.copy()
            cate_t, tstat_t, se_t = _double_ml_cate(X_conf, T_binary, Y_pae_f, method=cate_method, inner_n_jobs=1)
            elapsed = time.time() - t0
            # 详细特征贡献度统计
            n_pos = int(np.sum(cate_t > 0))
            n_neg = int(np.sum(cate_t < 0))
            n_sig = int(np.sum(np.abs(tstat_t) > 1.96))
            n_sig_pos = int(np.sum((tstat_t > 1.96)))
            n_sig_neg = int(np.sum((tstat_t < -1.96)))
            result = {
                'treatment': t_col, 'cate_mean': float(np.mean(cate_t)),
                'cate_std': float(np.std(cate_t)), 'cate_min': float(np.min(cate_t)),
                'cate_max': float(np.max(cate_t)), 'n_significant': n_sig,
                'pct_significant': float(np.mean(np.abs(tstat_t) > 1.96) * 100),
                'cate_median': float(np.median(cate_t)),
                'cate_p25': float(np.percentile(cate_t, 25)),
                'cate_p75': float(np.percentile(cate_t, 75)),
                'n_positive': n_pos, 'n_negative': n_neg,
                'n_sig_positive': n_sig_pos, 'n_sig_negative': n_sig_neg,
                'se_mean': float(np.mean(se_t)),
                'tstat_mean': float(np.mean(tstat_t)),
            }
            multi_treatment_results.append(result)
            print(f"  [{i+1}/{total}] {t_col}: cate_mean={result['cate_mean']:.3f} [{result['cate_p25']:.3f},{result['cate_p75']:.3f}], "
                  f"pos={n_pos} neg={n_neg} sig+={n_sig_pos} sig-={n_sig_neg}, SE={result['se_mean']:.4f}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{total}] {t_col}: 失败({e}), {elapsed:.1f}s")

    pd.DataFrame(multi_treatment_results).to_csv(out / 'multi_treatment_cate.csv', index=False)
    print(f"  完成: {len(multi_treatment_results)}/{total} 个treatment变量")

def _run_subgroup_discovery(feat_df, cate_pae, out, config):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    n_clusters = config.get("subgroup_n_clusters", 3)
    if cate_pae.std() > 1e-6:
        cate_2d = StandardScaler().fit_transform(cate_pae.reshape(-1, 1))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        subgroup_labels = km.fit_predict(cate_2d)
        np.save(out / 'subgroup_labels.npy', subgroup_labels)
        subgroup_profiles = []
        for sg in range(n_clusters):
            mask = subgroup_labels == sg
            sg_data = feat_df[mask]
            profile = {
                'subgroup': sg, 'size': int(mask.sum()),
                'rf2_pass_rate': float(sg_data['rf2_passed'].mean()) if 'rf2_passed' in sg_data.columns else 0,
                'final_candidate_rate': float(sg_data['final_candidate'].mean()) if 'final_candidate' in sg_data.columns else 0,
                'mean_pae': float(sg_data['rf2_interaction_pae'].mean()) if 'rf2_interaction_pae' in sg_data.columns else 0,
                'mean_cate': float(cate_pae[mask].mean()),
                'mean_cdr3_len': float(sg_data['cdr3_len'].mean()) if 'cdr3_len' in sg_data.columns else 0,
                'aromatic_ratio': float(sg_data['aromatic_ratio'].mean()) if 'aromatic_ratio' in sg_data.columns else 0,
                'glycine_ratio': float(sg_data['glycine_ratio'].mean()) if 'glycine_ratio' in sg_data.columns else 0,
            }
            subgroup_profiles.append(profile)
        pd.DataFrame(subgroup_profiles).to_csv(out / 'subgroup_profiles.csv', index=False)
        print(f"  亚群分析: {n_clusters}个亚群, 大小={[p['size'] for p in subgroup_profiles]}")

def _run_position_specific_cate(feat_df, embeddings, Y_pae, out, config):
    from sklearn.linear_model import Ridge
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
    return pos_cate_df

def _generate_counterfactual_suggestions(feat_df, failed_idx, passed_idx, cate_pae, se_pae,
                                           pos_cate_df, nn_indices, out, config):
    n_process = min(config.get("counterfactual_top_n", 2000), len(failed_idx))
    cf_suggestions = []
    for i in range(n_process):
        row = feat_df.iloc[failed_idx[i]]
        cdr3 = row['cdr3_sequence']
        if pd.isna(cdr3) or len(cdr3) == 0: continue
        seq_cate = cate_pae[failed_idx[i]]
        seq_se = se_pae[failed_idx[i]] if se_pae[failed_idx[i]] > 0 else 1.0
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
            is_significant = abs(seq_cate / seq_se) > 1.96 if seq_se > 0 else False
            cf_suggestions.append({
                'sequence_id': int(row['global_sequence_index']), 'original_cdr3': cdr3,
                'rank': rank+1, 'edit': f"Pos{s['pos']} {s['orig']}->{s['mut']}",
                'mutated_cdr3': s['cdr3'], 'predicted_pae_change': round(s['pae'], 2),
                'individual_cate': round(float(seq_cate), 3),
                'individual_se': round(float(seq_se), 3),
                'is_significant': is_significant,
                'edit_distance_to_template': sum(a!=b for a,b in zip(s['cdr3'], succ_cdr3)) if len(s['cdr3'])==len(succ_cdr3) else -1,
                'nearest_success_cdr3': succ_cdr3,
            })
    cf_df = pd.DataFrame(cf_suggestions)
    cf_df.to_csv(out / 'counterfactual_suggestions.csv', index=False)

def _generate_truncation_suggestions(feat_df, failed_idx, out, config):
    n_process = min(config.get("counterfactual_top_n", 2000), len(failed_idx))
    trunc_results = []
    for i in range(min(n_process, len(failed_idx))):
        row = feat_df.iloc[failed_idx[i]]
        cdr3 = row['cdr3_sequence']
        if pd.isna(cdr3) or len(cdr3) <= 7: continue
        for tl in [6, 7]:
            if len(cdr3) <= tl: continue
            trunc_results.append({'sequence_id': int(row['global_sequence_index']), 'original_cdr3': cdr3, 'original_len': len(cdr3), 'truncated_cdr3': cdr3[:tl], 'target_len': tl})
    pd.DataFrame(trunc_results).to_csv(out / 'truncation_suggestions.csv', index=False)

def _plot_tsne(embeddings, Y_binary, out, config):
    from sklearn.manifold import TSNE
    sample_size = min(config.get("tsne_sample_size", 3000), len(embeddings))
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeds_2d = tsne.fit_transform(embeddings[sample_idx])
    sample_labels = Y_binary[sample_idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(embeds_2d[sample_labels==0,0], embeds_2d[sample_labels==0,1], c='#e74c3c', alpha=0.3, s=10, label='Failed')
    ax.scatter(embeds_2d[sample_labels==1,0], embeds_2d[sample_labels==1,1], c='#2ecc71', alpha=0.6, s=20, label='Passed')
    ax.set_title('ESM-2 t-SNE'); ax.legend()
    plt.tight_layout(); plt.savefig(out / 'layer3_tsne_embedding.png', dpi=150, bbox_inches='tight'); plt.close()

# ═══════════════════════════════════════════════════════════════
# 阶段6: 规则合成与输出
# ═══════════════════════════════════════════════════════════════

def stage_layer5_synthesis(output_dir, config):
    out = Path(output_dir)
    feat_df = pd.read_csv(out / "feature_matrix.csv")

    target_label = config.get('optimization_target', 'final_candidate')
    target_col = 'final_candidate' if target_label == 'final_candidate' and 'final_candidate' in feat_df.columns else 'rf2_passed'

    len_pass = feat_df.groupby('cdr3_len').agg(
        total=(target_col, 'count'), rf2_pass=('rf2_passed', 'sum'), final_cand=('final_candidate', 'sum')
    ).reset_index()
    len_pass['pass_rate'] = len_pass['rf2_pass'] / len_pass['total']
    valid_lengths = len_pass[len_pass['pass_rate'] > 0.05]['cdr3_len'].tolist()

    fc_rate_by_len = feat_df.groupby('cdr3_len')[target_col].mean()
    dead_zone_lengths = [l for l in valid_lengths if fc_rate_by_len.get(l, 0) < 0.0005]
    if dead_zone_lengths:
        print(f"  ⚠ 移除FC死区长度(FC率≈0): {dead_zone_lengths}")
        valid_lengths = [l for l in valid_lengths if l not in dead_zone_lengths]

    first_res = feat_df.groupby('first_residue').agg(total=(target_col, 'count'), rf2_pass=('rf2_passed', 'sum')).reset_index()
    first_res['rate'] = first_res['rf2_pass'] / first_res['total']
    first_whitelist = first_res[first_res['rate'] > 0.70]['first_residue'].tolist()
    # 打印所有首残基的通过率排名
    print(f"  首残基RF2通过率排名(rate>0.70入选白名单):")
    for _, r in first_res.sort_values('rate', ascending=False).iterrows():
        mark = '✓' if r['rate'] > 0.70 else '✗'
        print(f"    {mark} {r['first_residue']}: {r['rate']:.1%} ({int(r['rf2_pass'])}/{int(r['total'])})")
    # 排除FC率为0的首残基（数据驱动：V等RF2通过率高但FC=0）
    fc_rate_by_first = feat_df.groupby('first_residue')[target_col].mean()
    zero_fc_first = [aa for aa in first_whitelist if fc_rate_by_first.get(aa, 0) == 0]
    if zero_fc_first:
        print(f"  ⚠ 移除FC率为0的首残基: {zero_fc_first} (RF2通过率高但从未成为最终候选)")
        first_whitelist = [aa for aa in first_whitelist if aa not in zero_fc_first]
    print(f"  首残基白名单: {first_whitelist}")

    last_res = feat_df.groupby('last_residue').agg(total=(target_col, 'count'), rf2_pass=('rf2_passed', 'sum')).reset_index()
    last_res['rate'] = last_res['rf2_pass'] / last_res['total']
    last_whitelist = last_res[last_res['rate'] > 0.08]['last_residue'].tolist()

    best_aromatic = _search_soft_threshold(feat_df, 'aromatic_ratio', '>=', [0.10, 0.15, 0.20, 0.25], target_col, 0.08, 0.15)
    best_glycine = _search_soft_threshold(feat_df, 'glycine_ratio', '<=', [0.25, 0.20, 0.15, 0.12], target_col, 0.08, 0.20)
    best_serine = _search_soft_threshold(feat_df, 'serine_ratio', '<=', [0.25, 0.20, 0.15, 0.10], target_col, 0.08, 0.15)
    best_hydrophobic = _search_soft_threshold(feat_df, 'hydrophobic_ratio', '>=', [0.35, 0.40, 0.45, 0.50, 0.55], target_col, 0.06, 0.40)

    anti_patterns = _detect_anti_patterns(feat_df, target_col)

    success_df = feat_df[feat_df[target_col] == True]
    success_templates = {}
    for length in sorted(set(valid_lengths)):
        len_s = success_df[success_df['cdr3_len'] == length]
        if len(len_s) == 0: continue
        success_templates[f'length_{length}'] = len_s['cdr3_sequence'].value_counts().head(10).index.tolist()

    salvage_edits = _extract_salvage_edits(out)
    trunc_to_7_count = _count_truncation_evidence(out)

    cox_hr = {}
    cox_path = out / 'cox_hazard_ratios.csv'
    if cox_path.exists():
        for _, r in pd.read_csv(cox_path).iterrows(): cox_hr[r['variable']] = round(r['HR'], 2)

    ate_vals = {}
    ate_path = out / 'ate_estimates.csv'
    if ate_path.exists():
        ate_df_file = pd.read_csv(ate_path)
        for _, r in ate_df_file.iterrows():
            if r['outcome'] == 'rf2_interaction_pae' and r['p_value'] < 0.05:
                ate_vals[r['treatment']] = round(r['ATE'], 2)

    strat_ate_map = {}
    strat_path = out / 'stratified_ate_estimates.csv'
    if strat_path.exists():
        strat_df = pd.read_csv(strat_path)
        for _, r in strat_df.iterrows():
            if r['p_value'] < 0.05:
                key = (int(r['cdr3_len']), r['treatment'])
                strat_ate_map[key] = r['ATE']

    length_specific_prefs = {}
    for length in valid_lengths:
        l_prefs = {}
        aro_ate = strat_ate_map.get((length, 'aromatic_ratio'))
        if aro_ate is not None:
            if aro_ate > 0:
                l_prefs['aromatic_min_ratio'] = 0.0
            elif aro_ate < -1.0:
                l_prefs['aromatic_min_ratio'] = best_aromatic
            else:
                l_prefs['aromatic_min_ratio'] = round(best_aromatic * 0.75, 3)
        else:
            l_prefs['aromatic_min_ratio'] = best_aromatic

        gly_ate = strat_ate_map.get((length, 'glycine_ratio'))
        if gly_ate is not None and gly_ate > 3.0:
            l_prefs['glycine_max_ratio'] = min(best_glycine, 0.15)
        else:
            l_prefs['glycine_max_ratio'] = best_glycine

        ser_ate = strat_ate_map.get((length, 'serine_ratio'))
        if ser_ate is not None and ser_ate < -3.0:
            l_prefs['serine_max_ratio'] = 0.4
        elif ser_ate is not None and ser_ate > 3.0:
            l_prefs['serine_max_ratio'] = min(best_serine, 0.12)
        else:
            l_prefs['serine_max_ratio'] = best_serine

        pro_ate = strat_ate_map.get((length, 'proline_count'))
        if pro_ate is not None and pro_ate < -3.0:
            l_prefs['proline_max_count'] = 3
        elif pro_ate is not None and pro_ate < 0:
            l_prefs['proline_max_count'] = 2
        else:
            l_prefs['proline_max_count'] = 1

        hyd_ate = strat_ate_map.get((length, 'hydrophobic_ratio'))
        if hyd_ate is not None:
            if hyd_ate > 1.0:
                l_prefs['hydrophobic_min_ratio'] = min(best_hydrophobic + 0.05, 0.60)
            elif hyd_ate > 0:
                l_prefs['hydrophobic_min_ratio'] = best_hydrophobic
            else:
                l_prefs['hydrophobic_min_ratio'] = round(best_hydrophobic * 0.80, 3)
        else:
            l_prefs['hydrophobic_min_ratio'] = best_hydrophobic

        first_aro_ate = strat_ate_map.get((length, 'first_is_aromatic'))
        if first_aro_ate is not None and first_aro_ate > -1.0:
            l_prefs['first_aromatic_optional'] = True

        length_specific_prefs[str(length)] = l_prefs

    length_weights = {}
    for _, row in len_pass.iterrows():
        l = int(row['cdr3_len'])
        if l in valid_lengths:
            length_weights[str(l)] = round(row['pass_rate'], 4)

    design_strategy = {
        'strategy_name': 'CSC-O_v2',
        'description': 'Causal-Stratified Counterfactual Optimization for Antibody CDR3 Design',
        'version': '2.0',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'data_source': str(config.get('input_file', '')),
        'data_size': len(feat_df),
        'optimization_target': target_label,
        'hard_constraints': {
            'cdr3_length_allowed': [int(l) for l in valid_lengths],
            'cdr3_length_preferred': [l for l in [6,7] if l in valid_lengths],
            'cdr3_first_residue_whitelist': first_whitelist,
            'cdr3_last_residue_whitelist': last_whitelist,
            'cdr3_min_positive_count': 0,
            'cdr3_min_aromatic_first': True,
        },
        'length_generation_weights': length_weights,
        'soft_preferences': {
            'aromatic_min_ratio': best_aromatic,
            'glycine_max_ratio': best_glycine,
            'serine_max_ratio': best_serine,
            'proline_max_count': 1,
            'hydrophobic_min_ratio': best_hydrophobic,
        },
        'length_specific_preferences': length_specific_prefs,
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

    _write_strategy_txt(out, valid_lengths, first_whitelist, last_whitelist,
                         best_aromatic, best_glycine, best_serine, best_hydrophobic,
                         anti_patterns, success_templates, salvage_edits)
    _write_analysis_report(out, config, feat_df, target_label, valid_lengths,
                            first_whitelist, last_whitelist, anti_patterns, salvage_edits)

    print(f"  策略文件: design_strategy.json, design_strategy.txt, csco_analysis_report.txt")
    return {"valid_lengths": valid_lengths, "anti_patterns": anti_patterns, "n_salvage_edits": len(salvage_edits)}

def _search_soft_threshold(feat_df, col, direction, thresholds, target_col, min_rate, default):
    for t in thresholds:
        if direction == '>=':
            mask = feat_df[col] >= t
        else:
            mask = feat_df[col] <= t
        if mask.sum() > 0:
            rate = feat_df.loc[mask, target_col].mean()
            if rate > min_rate:
                return t
    return default

def _detect_anti_patterns(feat_df, target_col):
    anti_patterns = []
    for pattern, col in [('GGG', 'has_ggg'), ('SSS', 'has_sss'), ('LL', 'has_ll')]:
        if col in feat_df.columns and feat_df[col].sum() > 0:
            with_rate = feat_df.loc[feat_df[col] == 1, target_col].mean()
            without_rate = feat_df.loc[feat_df[col] == 0, target_col].mean()
            if without_rate > 0 and with_rate < without_rate * 0.3:
                anti_patterns.append(pattern)
    return anti_patterns

def _extract_salvage_edits(out):
    salvage_edits = []
    cf_path = out / 'counterfactual_suggestions.csv'
    if cf_path.exists():
        cf_df = pd.read_csv(cf_path)
        for pattern, count in cf_df['edit'].value_counts().head(30).items():
            mean_pae = cf_df[cf_df['edit'] == pattern]['predicted_pae_change'].mean()
            if mean_pae < -0.5:
                salvage_edits.append({'pattern': pattern, 'count': int(count), 'mean_pae_change': round(mean_pae, 2)})
    return salvage_edits

def _count_truncation_evidence(out):
    trunc_path = out / 'truncation_suggestions.csv'
    if trunc_path.exists():
        trunc_df = pd.read_csv(trunc_path)
        return len(trunc_df[(trunc_df['original_len'] >= 10) & (trunc_df['target_len'] == 7)])
    return 0

def _write_strategy_txt(out, valid_lengths, first_whitelist, last_whitelist,
                          best_aromatic, best_glycine, best_serine, best_hydrophobic,
                          anti_patterns, success_templates, salvage_edits):
    lines = ['# CSC-O Design Strategy', f'# Generated: {datetime.now().strftime("%Y-%m-%d")}', '']
    lines.append(f'CDR3_LENGTH_ALLOWED = {[int(l) for l in valid_lengths]}')
    lines.append(f'CDR3_FIRST_RESIDUE_WHITELIST = {first_whitelist}')
    lines.append(f'CDR3_LAST_RESIDUE_WHITELIST = {last_whitelist}')
    lines.append(f'AROMATIC_MIN_RATIO = {best_aromatic}')
    lines.append(f'GLYCINE_MAX_RATIO = {best_glycine}')
    lines.append(f'SERINE_MAX_RATIO = {best_serine}')
    lines.append(f'HYDROPHOBIC_MIN_RATIO = {best_hydrophobic}')
    lines.append(f'PROLINE_MAX_COUNT = 1')
    for ap in anti_patterns: lines.append(f'FORBIDDEN_PATTERN = {ap}')
    for lk, tmpls in success_templates.items():
        for t in tmpls: lines.append(f'TEMPLATE_{lk.upper()} = {t}')
    for se in salvage_edits[:15]: lines.append(f'SALVAGE_EDIT = {se["pattern"]}  # count={se["count"]}, pae={se["mean_pae_change"]}')
    lines.append('TRUNCATION_RULE = IF_LEN_GE_10_THEN_TRUNCATE_TO_7')
    with open(out / 'design_strategy.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def _write_analysis_report(out, config, feat_df, target_label, valid_lengths,
                             first_whitelist, last_whitelist, anti_patterns, salvage_edits):
    target_col = 'final_candidate' if target_label == 'final_candidate' else 'rf2_passed'
    report = ['='*80, 'CSC-O 综合分析报告', '='*80, '']
    report.append(f'数据来源: {config.get("input_file", "")}')
    report.append(f'数据量: {len(feat_df)} 条')
    report.append(f'优化目标: {target_label}')
    report.append(f'RF2通过率: {feat_df["rf2_passed"].mean()*100:.1f}%')
    report.append(f'最终候选率: {feat_df["final_candidate"].mean()*100:.2f}%')
    if target_label == 'final_candidate':
        report.append(f'Final candidate正样本数: {int(feat_df["final_candidate"].sum())}')
    report.append(f'CDR3长度允许值: {valid_lengths}')
    report.append(f'首残基白名单: {first_whitelist}')
    report.append(f'尾残基白名单: {last_whitelist}')
    report.append(f'反模式: {anti_patterns}')
    report.append(f'高频挽救编辑: {[se["pattern"] for se in salvage_edits[:5]]}')
    report.append('='*80)
    with open(out / 'csco_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

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
                adapter = DataAdapter(config["input_file"], config=config)
                df = adapter.validate()
                metrics = stage_data_engineering(df, output_dir, config)
            else:
                metrics = STAGE_MAP[stage_name](output_dir, config)
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
    parser.add_argument("--esm2-models", type=str, default=None, help="ESM-2模型列表(逗号分隔)")
    parser.add_argument("--esm2-fusion", type=str, default="concat", choices=["concat","average","pca_concat"])
    parser.add_argument("--esm2-pca-dim", type=int, default=256, help="PCA降维目标维度")
    parser.add_argument("--cate-method", type=str, default="causal_forest", choices=["causal_forest","r_learner","plr"])
    parser.add_argument("--subgroup-clusters", type=int, default=3, help="亚群聚类数")
    parser.add_argument("--target", "-t", type=str, default="final_candidate",
                        choices=["final_candidate", "rf2_passed"],
                        help="优化目标: final_candidate 或 rf2_passed")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["input_file"] = args.input
    config["output_dir"] = args.output
    config["work_dir"] = args.work
    config["stage_resume"] = args.resume
    config["device"] = args.device
    config["esm2_batch_size"] = args.batch_size
    config["counterfactual_top_n"] = args.top_n
    config["esm2_fusion"] = args.esm2_fusion
    config["esm2_pca_dim"] = args.esm2_pca_dim
    config["cate_method"] = args.cate_method
    config["subgroup_n_clusters"] = args.subgroup_clusters
    config["optimization_target"] = args.target
    if args.esm2_models:
        config["esm2_models"] = [m.strip() for m in args.esm2_models.split(",")]
        config["esm2_model"] = config["esm2_models"][0]

    run_pipeline(config)
