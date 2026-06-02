import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = './Q02223_first50_all_sequences(1).csv'
OUTPUT_DIR = './output'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

from csco_config import AMINO_ACIDS, AROMATIC, POSITIVE, NEGATIVE, HYDROPHOBIC, GLYCINE, SERINE, PROLINE, extract_cdr3_features

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def build_feature_matrix(df):
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

        if row['rf2_passed_filter']:
            feats['survival_time'] = 1
        else:
            feats['survival_time'] = 0

        if row['funnel_stage'] == 'rf2_failed':
            feats['survival_event'] = 1
            feats['death_stage'] = 1
        elif row['funnel_stage'] == 'rf2_passed':
            feats['survival_event'] = 0
            feats['death_stage'] = 0
        elif row['funnel_stage'] == 'final_candidate':
            feats['survival_event'] = 0
            feats['death_stage'] = 0
        else:
            feats['survival_event'] = 0
            feats['death_stage'] = 0

        feats['af3_analyzed'] = row['af3_analyzed']
        feats['schrodinger_analyzed'] = row['schrodinger_analyzed']
        feats['af3_passed'] = row.get('af3_passed_filter', False)
        feats['schrodinger_passed'] = row.get('schrodinger_passed_filter', False)

        if pd.notna(row.get('af3_iptm')):
            feats['af3_iptm'] = row['af3_iptm']
        else:
            feats['af3_iptm'] = np.nan

        if pd.notna(row.get('mmgbsa_delta_g')):
            feats['mmgbsa_delta_g'] = row['mmgbsa_delta_g']
        else:
            feats['mmgbsa_delta_g'] = np.nan

        feats['rf2_filter_reason'] = row.get('rf2_filter_reason', '')

        features_list.append(feats)

    feat_df = pd.DataFrame(features_list)
    return feat_df

def build_survival_data(feat_df):
    survival_records = []
    for _, row in feat_df.iterrows():
        stage_times = {
            1: row['rf2_passed'],
            2: row['af3_analyzed'],
            3: row['schrodinger_analyzed'],
            4: row['final_candidate'],
        }

        if not stage_times[1]:
            t = 1
            event = 1
        elif not stage_times[2]:
            t = 2
            event = 1
        elif not stage_times[3]:
            t = 3
            event = 1
        elif not stage_times[4]:
            t = 4
            event = 1
        else:
            t = 4
            event = 0

        survival_records.append({
            'global_sequence_index': row['global_sequence_index'],
            'time': t,
            'event': event,
            'cdr3_len': row['cdr3_len'],
            'positive_ratio': row['positive_ratio'],
            'aromatic_ratio': row['aromatic_ratio'],
            'glycine_ratio': row['glycine_ratio'],
            'serine_ratio': row['serine_ratio'],
            'proline_count': row['proline_count'],
            'hydrophobic_ratio': row['hydrophobic_ratio'],
            'negative_ratio': row['negative_ratio'],
            'first_residue': row['first_residue'],
            'last_residue': row['last_residue'],
            'backbone_id': row['backbone_id'],
            'rf2_interaction_pae': row['rf2_interaction_pae'],
            'rf2_pred_lddt': row['rf2_pred_lddt'],
        })

    return pd.DataFrame(survival_records)

if __name__ == '__main__':
    df = load_data()
    feat_df = build_feature_matrix(df)
    surv_df = build_survival_data(feat_df)

    feat_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index=False)
    surv_df.to_csv(os.path.join(OUTPUT_DIR, 'survival_data.csv'), index=False)

    print(f'Feature matrix shape: {feat_df.shape}')
    print(f'Survival data shape: {surv_df.shape}')
    print(f'Saved to {OUTPUT_DIR}')
