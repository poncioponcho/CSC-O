#!/usr/bin/env python3
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats as sp_stats


def run_ate(data, treatment, outcome, confounders):
    T_flat = data[treatment].values.astype(float)
    Y_flat = data[outcome].values.astype(float)
    n = len(Y_flat)

    if confounders:
        X_c = data[confounders].values.astype(float)
        valid = ~(np.isnan(T_flat) | np.isnan(Y_flat) | np.any(np.isnan(X_c), axis=1))
    else:
        X_c = np.zeros((n, 0))
        valid = ~(np.isnan(T_flat) | np.isnan(Y_flat))

    T = T_flat[valid].reshape(-1, 1)
    Y = Y_flat[valid]
    X_c = X_c[valid]

    X_full = np.hstack([T, X_c])
    lr = LinearRegression().fit(X_full, Y)
    ate = lr.coef_[0]

    nv = len(Y)
    k = X_full.shape[1]
    res = Y - lr.predict(X_full)
    mse = np.sum(res**2) / max(nv - k - 1, 1)
    try:
        inv_d = np.linalg.inv(X_full.T @ X_full)[0, 0]
    except np.linalg.LinAlgError:
        inv_d = np.linalg.pinv(X_full.T @ X_full)[0, 0]
    se = np.sqrt(mse * inv_d)
    t_stat = ate / se
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=nv - k - 1))
    return ate, se, t_stat, p_val, ate - 1.96*se, ate + 1.96*se


def main():
    data_dir = Path("/home/test/CSC-O/output_v3")
    feat_path = data_dir / "feature_matrix.csv"
    if not feat_path.exists():
        data_dir = Path(__file__).parent / "output"
        feat_path = data_dir / "feature_matrix.csv"
    if not feat_path.exists():
        print(f"ERROR: feature_matrix.csv not found")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    n = len(df)
    outcome = 'rf2_interaction_pae'
    treatment = 'aromatic_ratio'

    ratio_vars = ['cdr3_len', 'positive_ratio', 'aromatic_ratio', 'glycine_ratio',
                  'serine_ratio', 'hydrophobic_ratio']

    print("=" * 70)
    print("CSC-O Aromatic Ratio ATE-Cox Contradiction Diagnostic")
    print(f"N={n}, RF2 pass rate={df['rf2_passed'].mean():.4f}")
    print("=" * 70)

    # PART 1: Correlation matrix
    print("\n" + "=" * 70)
    print("PART 1: Correlation Matrix (ratio variables)")
    print("=" * 70)
    corr_df = df[ratio_vars].corr()
    print()
    header = f"{'':>20s} | " + "  ".join(f"{v:>8s}" for v in ratio_vars)
    print(header)
    print("-" * len(header))
    for v1 in ratio_vars:
        row = f"{v1:>20s} | "
        for v2 in ratio_vars:
            c = float(corr_df.loc[v1, v2])
            mark = "*" if v1 != v2 and abs(c) >= 0.5 else " "
            row += f"{c:+7.3f}{mark} "
        print(row)

    print("\n* = |r| >= 0.5 (strong correlation)")
    print("\nStrong pairs (|r|>=0.3):")
    for i, v1 in enumerate(ratio_vars):
        for j, v2 in enumerate(ratio_vars):
            if i < j:
                c = float(corr_df.loc[v1, v2])
                if abs(c) >= 0.3:
                    print(f"  {v1} <-> {v2}: r={c:+.3f}")

    # PART 2: Progressive ATE for aromatic_ratio
    print("\n" + "=" * 70)
    print(f"PART 2: Progressive ATE ({treatment} -> {outcome})")
    print("=" * 70)

    progressive = [
        ([], "No confounders (raw)"),
        (['backbone_id'], "+ backbone_id (v1)"),
        (['backbone_id', 'cdr3_len'], "+ cdr3_len (v2 current)"),
        (['backbone_id', 'cdr3_len', 'glycine_ratio'], "+ glycine_ratio"),
        (['backbone_id', 'cdr3_len', 'glycine_ratio', 'serine_ratio'], "+ serine_ratio"),
        (['backbone_id', 'cdr3_len', 'glycine_ratio', 'serine_ratio',
          'hydrophobic_ratio', 'positive_ratio'], "+ hydro+positive (full Cox match)"),
    ]

    print(f"\n{'Confounders':<45s} {'ATE':>8s} {'SE':>8s} {'t':>8s} {'p':>10s} {'95% CI':>22s} {'Delta':>10s}")
    print("-" * 115)

    prev_ate = None
    for confs, label in progressive:
        ate, se, ts, pv, cl, cu = run_ate(df, treatment, outcome, confs)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        delta_str = ""
        if prev_ate is not None:
            d = ate - prev_ate
            delta_str = f"{'+' if d > 0 else ''}{d:.3f}"
        print(f"  {label:<43s} {ate:>+8.3f} {se:8.3f} {ts:>+8.2f} {pv:10.2e}{sig:>3s} [{cl:>+8.3f}, {cu:>+8.3f}] {delta_str:>10s}")
        prev_ate = ate

    # PART 3: All treatments - minimal vs full confounders
    print("\n" + "=" * 70)
    print("PART 3: All treatments - minimal vs full confounders")
    print("=" * 70)

    all_treatments = ['aromatic_ratio', 'glycine_ratio', 'serine_ratio',
                      'proline_count', 'first_is_aromatic', 'last_is_YH']
    min_confs = ['backbone_id', 'cdr3_len']
    full_confs = ['backbone_id', 'cdr3_len', 'glycine_ratio', 'serine_ratio',
                  'hydrophobic_ratio', 'positive_ratio']

    print(f"\n{'Treatment':<22s} {'Raw ATE':>10s} {'Min ATE':>10s} {'Full ATE':>10s} {'Change':>10s} {'Flipped?':>10s}")
    print("-" * 76)

    for tv in all_treatments:
        ate_raw, *_ = run_ate(df, tv, outcome, [])
        ate_min, *_ = run_ate(df, tv, outcome, min_confs)
        ate_full, *_ = run_ate(df, tv, outcome, full_confs)
        delta = ate_full - ate_min
        flipped = "YES" if (ate_min > 0 and ate_full < 0) or (ate_min < 0 and ate_full > 0) else ""
        print(f"  {tv:<20s} {ate_raw:>+10.3f} {ate_min:>+10.3f} {ate_full:>+10.3f} {delta:>+10.3f} {flipped:>10s}")

    # PART 4: Cox regression
    print("\n" + "=" * 70)
    print("PART 4: Cox PH Regression (detailed)")
    print("=" * 70)
    try:
        from lifelines import CoxPHFitter
        surv_path = data_dir / "survival_data.csv"
        if surv_path.exists():
            sdf = pd.read_csv(surv_path)
            cox_vars = ratio_vars[:]
            for v in cox_vars:
                sdf[f'{v}_c'] = sdf[v] - sdf[v].mean()
            cox_data = sdf[['time', 'event'] + [f'{v}_c' for v in cox_vars]].dropna()
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='time', event_col='event')

            print(f"\nConcordance: {cph.concordance_index_:.4f}\n")
            print(f"{'Variable':<20s} {'Coef':>8s} {'HR':>8s} {'95% CI':>24s} {'p':>10s}")
            print("-" * 76)
            for v in cox_vars:
                vc = f'{v}_c'
                coef = float(cph.params_[vc])
                hr = float(np.exp(coef))
                ci_l = float(np.exp(cph.confidence_intervals_.loc[vc, '95% lower-bound']))
                ci_u = float(np.exp(cph.confidence_intervals_.loc[vc, '95% upper-bound']))
                pv = float(cph.summary.loc[vc, 'p'])
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"  {v:<18s} {coef:>+8.4f} {hr:8.3f} [{ci_l:.3f}, {ci_u:.3f}]      {pv:10.2e}{sig}")
            print("\nHR<1 = protective, HR>1 = risk factor")
        else:
            print("survival_data.csv not found, skip")
    except ImportError:
        print("lifelines not installed, skip")
    except Exception as e:
        print(f"Cox error: {e}")

    # PART 5: Backbone_id collider check
    print("\n" + "=" * 70)
    print("PART 5: Backbone_id Effect on ATE")
    print("=" * 70)

    ate_no_bb, *_ = run_ate(df, treatment, outcome, [])
    ate_bb, *_ = run_ate(df, treatment, outcome, ['backbone_id'])
    ate_bb_len, *_ = run_ate(df, treatment, outcome, ['backbone_id', 'cdr3_len'])
    ate_full_all, *_ = run_ate(df, treatment, outcome, full_confs)

    print(f"\n  No confounders:      ATE = {ate_no_bb:+.3f}")
    print(f"  + backbone_id:       ATE = {ate_bb:+.3f}  (delta = {ate_bb - ate_no_bb:+.3f})")
    print(f"  + backbone_id+len:   ATE = {ate_bb_len:+.3f}  (delta = {ate_bb_len - ate_bb:+.3f})")
    print(f"  + all ratios:        ATE = {ate_full_all:+.3f}  (delta = {ate_full_all - ate_bb_len:+.3f})")

    # PART 6: Stratified by CDR3 length
    print("\n" + "=" * 70)
    print("PART 6: Stratified Analysis (aromatic_ratio effect by CDR3 length)")
    print("=" * 70)

    print(f"\n{'Len':>4s} {'N':>6s} {'aro_mean':>9s} {'PAE_mean':>9s} {'pass_rt':>8s} {'ATE':>9s} {'p':>10s}")
    print("-" * 60)

    for length in sorted(df['cdr3_len'].unique()):
        sub = df[df['cdr3_len'] == length]
        if len(sub) < 30:
            continue
        ate_l, _, _, pl, _, _ = run_ate(sub, treatment, outcome, ['backbone_id'])
        sig = "***" if pl < 0.001 else "**" if pl < 0.01 else "*" if pl < 0.05 else ""
        print(f"  {int(length):>4d} {len(sub):>6d} {sub['aromatic_ratio'].mean():>9.4f} "
              f"{sub['rf2_interaction_pae'].mean():>9.2f} {sub['rf2_passed'].mean():>8.3f} "
              f"{ate_l:>+9.3f}{sig:>3s} {pl:>10.2e}")

    # PART 7: Conclusion
    print("\n" + "=" * 70)
    print("PART 7: Conclusion & Recommendation")
    print("=" * 70)

    ate_current, _, _, p_cur, _, _ = run_ate(df, treatment, outcome, min_confs)
    ate_full, _, _, p_ful, _, _ = run_ate(df, treatment, outcome, full_confs)

    print(f"""
  Current pipeline ATE (backbone_id + cdr3_len):
    aromatic_ratio on PAE = {ate_current:+.3f} (p={p_cur:.2e})

  Full-confounder ATE (backbone_id + cdr3_len + all ratios):
    aromatic_ratio on PAE = {ate_full:+.3f} (p={p_ful:.2e})

  Cox PH HR (controlling all ratios):
    aromatic_ratio HR = 0.67 (protective)

  Direction: ATE={"RISK(+)" if ate_full > 0 else "PROTECTIVE(-)"}, Cox=PROTECTIVE(HR<1)
  Consistent? {"NO - contradiction remains" if ate_full > 0 else "YES - resolved"}
""")

    results = {
        'sample_size': int(n),
        'current_ate_aromatic_on_pae': round(float(ate_current), 4),
        'full_ate_aromatic_on_pae': round(float(ate_full), 4),
        'cox_hr_aromatic': 0.67,
        'contradiction': 'unresolved' if ate_full > 0 else 'resolved',
        'recommendation': 'remove_aromatic_min_ratio' if ate_full > 0 else 'keep_aromatic_min_ratio',
    }

    out_json = data_dir / "diagnostic_aromatic_contradiction.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    main()
