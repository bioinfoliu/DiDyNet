import os
from collections import defaultdict
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from dtaidistance import dtw
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

def extract_filtered_features_df(df, filtered_features, id_col="SubjectID", time_col="Time"):
    base_cols = [id_col, time_col]
    feats = [c for c in filtered_features if c in df.columns]
    return df.loc[:, base_cols + feats].copy()

def get_features_for_k(results_one_omic: dict, k: int, mode: str):
    entry = results_one_omic.get(int(k), {})
    return entry.get(mode, [])

def build_omics_by_k(data_map: dict, results_map: dict, k: int, mode: str, id_col="SubjectID", time_col="Time"):
    omics_by_k = {}
    for name, df in data_map.items():
        if name not in results_map:
            raise KeyError(f"results_map is missing {name}")
        feats = get_features_for_k(results_map[name], k, mode)
        omics_by_k[name] = extract_filtered_features_df(df, feats, id_col=id_col, time_col=time_col)
    return omics_by_k

def preprocess_dataframe_to_dict(df, id_col="SubjectID", time_col="Time"):
    df_sorted = df.sort_values(by=[id_col, time_col])
    features = [c for c in df.columns if c not in [id_col, time_col]]
    
    data_dict = defaultdict(dict)
    for subj, group in df_sorted.groupby(id_col):
        for f in features:
            vals = group[f].dropna().values.astype(float)
            if len(vals) >= 2:
                data_dict[subj][f] = vals
                
    return dict(data_dict), features

def compute_dtw_for_feature_pair(f1, f2, dict1, dict2, subjects):
    results = []
    for subj in subjects:
        s1 = dict1.get(subj, {}).get(f1)
        s2 = dict2.get(subj, {}).get(f2)
        
        if s1 is not None and s2 is not None:
            try:
                dist = dtw.distance(s1, s2)
                results.append((f1, f2, subj, dist))
            except Exception:
                pass 
                
    if results:
        return results
    return None

def run_wilcoxon_test_in_memory(df_distances, out_file):
    if df_distances.empty:
        print("⚠️ Distance matrix is empty, skipping test.")
        return

    df_distances['pair'] = df_distances.apply(lambda row: tuple(sorted([row['feature1'], row['feature2']])), axis=1)

    results = []
    for pair, group_df in df_distances.groupby('pair'):
        group_IS = group_df[group_df['group'] == 'IS']['distance']
        group_IR = group_df[group_df['group'] == 'IR']['distance']

        if len(group_IS) < 3 or len(group_IR) < 3:
            continue
        
        stat, pval = ranksums(group_IS, group_IR)
        results.append({
            'feature1': pair[0],
            'feature2': pair[1],
            'p_value': pval,
            'mean_IS': group_IS.mean(),
            'mean_IR': group_IR.mean(),
            'mean_diff': group_IS.mean() - group_IR.mean()
        })

    pval_df = pd.DataFrame(results)

    if not pval_df.empty:
        _, pval_df['adj_p'], _, _ = multipletests(pval_df['p_value'], alpha=0.05, method='fdr_bh')
        pval_df['significant'] = pval_df['adj_p'] < 0.05
    else:
        pval_df['adj_p'] = []
        pval_df['significant'] = []

    pval_df.to_csv(out_file, index=False)
    print(f"✅ Test complete: {os.path.basename(out_file)} | Significant feature pairs found (FDR<0.05): {pval_df['significant'].sum()}")

def compute_groupwise_dtw_and_test(
    omics1, omics2, subjects, omics1_name, omics2_name, group_name,
    IRIS_label, DTW_path, tag="exp", id_col="SubjectID", time_col="Time"
):
    dict1, feats1 = preprocess_dataframe_to_dict(omics1, id_col, time_col)
    dict2, feats2 = preprocess_dataframe_to_dict(omics2, id_col, time_col)

    is_intra = (omics1_name == omics2_name)
    if is_intra:
        feature_pairs = [(feats1[i], feats1[j]) for i in range(len(feats1)) for j in range(i, len(feats1))]
    else:
        feature_pairs = [(a, b) for a in feats1 for b in feats2]

    print(f"\n🧬 Starting comparison: {omics1_name} × {omics2_name} | Valid feature pairs generated: {len(feature_pairs)}")

    parallel_outputs = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(compute_dtw_for_feature_pair)(f1, f2, dict1, dict2, subjects)
        for f1, f2 in tqdm(feature_pairs, desc=f"🚀 Calculating DTW distances")
    )

    all_sims = []
    for out in parallel_outputs:
        if out is not None:
            all_sims.extend(out)

    all_sims_df = pd.DataFrame(all_sims, columns=["feature1", "feature2", "subject", "distance"])
    
    if not all_sims_df.empty:
        all_sims_df = all_sims_df.merge(IRIS_label[[id_col, 'IRIS']], left_on='subject', right_on=id_col, how='left')
        all_sims_df = all_sims_df.rename(columns={'IRIS': 'group'}).drop(columns=[id_col])
        all_sims_df = all_sims_df.dropna(subset=['group'])

    raw_filename = f"{omics1_name}_{omics2_name}_{group_name}_{tag}_all_distances_with_group_dedup.csv"
    raw_path = os.path.join(DTW_path, raw_filename)
    all_sims_df.to_csv(raw_path, index=False)
    print(f"✅ Distance matrix saved to: {raw_filename}")
    
    test_filename = f"{omics1_name}_{omics2_name}_{group_name}_{tag}_wilcoxon_results.csv"
    test_path = os.path.join(DTW_path, test_filename)
    run_wilcoxon_test_in_memory(all_sims_df, test_path)

def run_all_combinations_for_k(omics_by_k: dict, subjects, IRIS_label, k: int, mode: str,
                               DTW_path="results/DTW", group_name="IS_vs_IR",
                               time_col="Time", id_col="SubjectID"):
    os.makedirs(DTW_path, exist_ok=True)
    omics_names = list(omics_by_k.keys())
    pairs = [(omics_names[i], omics_names[j]) for i in range(len(omics_names)) for j in range(i, len(omics_names))]
    tag = f"k{k}_{mode}"

    for (n1, n2) in pairs:
        compute_groupwise_dtw_and_test(
            omics1=omics_by_k[n1], omics2=omics_by_k[n2],
            subjects=subjects, omics1_name=n1, omics2_name=n2,
            group_name=group_name, IRIS_label=IRIS_label,
            DTW_path=DTW_path, tag=tag,
            time_col=time_col, id_col=id_col
        )

def run_dtw_pipeline(data_map, results_map, IRIS_label, subjects_all, base_path, target_k):
    print("\n====================================")
    print("=== Step 3: DTW Calculation and Significance Testing ===")
    print("====================================")
    DTW_path = os.path.join(base_path, "results/DTW")
    os.makedirs(DTW_path, exist_ok=True)

    mode = "union"   
    id_col, time_col = "SubjectID", "Time"

    print(f"\n>>>> Starting pipeline for K={target_k} <<<<")
    omics_k = build_omics_by_k(data_map, results_map, k=target_k, mode=mode, id_col=id_col, time_col=time_col)

    run_all_combinations_for_k(
        omics_by_k=omics_k,
        subjects=subjects_all,
        IRIS_label=IRIS_label,
        k=target_k, mode=mode,
        DTW_path=DTW_path,
        group_name="IS_vs_IR",
        time_col=time_col, id_col=id_col
    )