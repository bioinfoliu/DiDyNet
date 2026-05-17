import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
import os
import glob
from pathlib import Path
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import time

LMM_ALPHA = 0.05 

def load_and_prepare_data(base_path):
    print("📂 Reading cleaned data files...")
    clean_data_dir = os.path.join(base_path, "data/cleaned")
    try:
        cyto_df = pd.read_csv(os.path.join(clean_data_dir, "cytokines_cleaned.csv"))
        prot_df = pd.read_csv(os.path.join(clean_data_dir, "proteomics_cleaned.csv"))
        trans_df = pd.read_csv(os.path.join(clean_data_dir, "transcriptome_cleaned.csv"))
        label_df = pd.read_csv(os.path.join(clean_data_dir, "IRIS_label_cleaned.csv"))
    except FileNotFoundError as e:
        print(f"❌ Error: File not found! {e}")
        return None

    shared_subjects = set(cyto_df['SubjectID']).intersection(
        set(prot_df['SubjectID'])
    ).intersection(
        set(trans_df['SubjectID'])
    )
    shared_list = sorted(list(shared_subjects))
    print(f"🎯 Identified {len(shared_list)} shared subjects (Intersection).")

    iris_map = label_df[['SubjectID', 'IRIS']].drop_duplicates().set_index('SubjectID')['IRIS'].to_dict()

    def to_long(df, name):
        df_sub = df[df['SubjectID'].isin(shared_list)].copy()
        id_vars = ['SubjectID', 'Time']
        feats = [c for c in df_sub.columns if c not in id_vars]
        
        df_long = pd.melt(df_sub, id_vars=id_vars, value_vars=feats, 
                          var_name='Feature', value_name='Value')
        
        df_long['Group'] = df_long['SubjectID'].map(iris_map)
        return df_long.rename(columns={'SubjectID': 'Subject'})

    raw_data_map = {
        "cytokines": to_long(cyto_df, "cytokines"),
        "proteomics": to_long(prot_df, "proteomics"),
        "transcriptome": to_long(trans_df, "transcriptome")
    }
    return raw_data_map

def parse_filename_safe(fn: str):
    stem = fn.replace(".csv", "")
    parts = stem.split("_")
    k_val, mode, om1, om2 = None, None, parts[0], parts[1]
    for p in parts:
        if p.startswith("k") and p[1:].isdigit():
            k_val = int(p[1:])
        if p in ["union", "intersection"]:
            mode = p
    return om1, om2, k_val, mode

def get_required_features(base_path, target_k):
    print("🔍 Scanning DTW result files, extracting required feature subsets...")
    dtw_results_dir = os.path.join(base_path, "results/DTW")
    files = glob.glob(os.path.join(dtw_results_dir, "*_wilcoxon_results.csv"))
    
    required_features = {"cytokines": set(), "proteomics": set(), "transcriptome": set()}
    valid_files = []
    
    for fp in files:
        fn = Path(fp).name
        parsed = parse_filename_safe(fn)
        if not parsed: continue
        om1, om2, K, mode = parsed
        
        if mode != "union" or K != target_k: continue
            
        df = pd.read_csv(fp)
        sig_col = next((c for c in ["signif_bonf_0.05", "significant"] if c in df.columns), None)
        if not sig_col: continue
            
        df_sig = df[df[sig_col] == True]
        if not df_sig.empty:
            f1_col = 'feature1' if 'feature1' in df.columns else 'feature_g'
            f2_col = 'feature2' if 'feature2' in df.columns else 'feature_h'
            
            if om1 in required_features and f1_col in df_sig.columns:
                required_features[om1].update(df_sig[f1_col].dropna().unique())
            if om2 in required_features and f2_col in df_sig.columns:
                required_features[om2].update(df_sig[f2_col].dropna().unique())
                
        valid_files.append((fp, fn, om1, om2, K, mode, sig_col))
        
    for om, feats in required_features.items():
        print(f"   - {om}: Extracted {len(feats)} features required for calculation")
        
    return required_features, valid_files

def run_lmm_for_selected_features(df_long, omics_name, target_features, save_dir):
    cache_file = os.path.join(save_dir, f"lmm_pvals_{omics_name}_target_only.csv")
    
    if os.path.exists(cache_file):
        print(f"✅ [{omics_name}] Loading cached target feature P-values...")
        tmp = pd.read_csv(cache_file, index_col=0)
        return tmp['p_value'].to_dict()

    feats_to_run = [f for f in target_features if f in df_long["Feature"].unique()]
    print(f"🚀 [{omics_name}] Starting LMM calculation (only for the extracted {len(feats_to_run)} target features)...")
    
    p_values = {}
    warnings.simplefilter('ignore', ConvergenceWarning)
    
    for feat in tqdm(feats_to_run, desc=omics_name):
        sub_df = df_long[df_long["Feature"] == feat].copy()
        if len(sub_df.dropna()) < 10 or sub_df["Value"].nunique() <= 1:
            p_values[feat] = 1.0
            continue
            
        try:
            model = smf.mixedlm("Value ~ Group * Time", sub_df, groups=sub_df["Subject"])
            result = model.fit(reml=False)
            term = [x for x in result.pvalues.index if ":" in x]
            p_values[feat] = result.pvalues[term[0]] if term else 1.0
        except:
            p_values[feat] = 1.0
            
    warnings.simplefilter('default', ConvergenceWarning)
    pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value']).to_csv(cache_file)
    return p_values

def classify_ddc_pairs(pairs_df, p_dict1, p_dict2, f1_col, f2_col):
    categories = []
    for _, row in pairs_df.iterrows():
        f1 = row.get(f1_col)
        f2 = row.get(f2_col)
        
        p1 = p_dict1.get(f1, 1.0)
        p2 = p_dict2.get(f2, 1.0)
        
        sig1 = p1 < LMM_ALPHA
        sig2 = p2 < LMM_ALPHA
        
        if not sig1 and not sig2:
            cat = "A_Subtle_Coordinated"
        elif sig1 and sig2:
            cat = "C_Both_Driven"
        else:
            cat = "B_Unilateral_Driver"
        categories.append(cat)
        
    pairs_df['Category'] = categories
    return pairs_df

def run_post_hoc(base_path, target_k):
    print("\n====================================")
    print("=== Step 4: LMM Post-hoc Refinement and Classification ===")
    print("====================================")
    global_start_time = time.time() 
    save_dir = os.path.join(base_path, "results/PostHoc")
    os.makedirs(save_dir, exist_ok=True)

    required_features, valid_files = get_required_features(base_path, target_k)
    if not valid_files:
        print("❌ No matching DTW result files found, please check the directory and filter conditions!")
        return

    raw_data_map = load_and_prepare_data(base_path)
    if not raw_data_map: return

    lmm_cache = {}
    print("\n--- Step 1: Calculate LMM background significance for target features ---")
    lmm_start_time = time.time() 
    for omics, df_long in raw_data_map.items():
        if required_features[omics]: 
            lmm_cache[omics] = run_lmm_for_selected_features(df_long, omics, required_features[omics], save_dir)
        else:
            lmm_cache[omics] = {}
    lmm_end_time = time.time() 

    print(f"\n--- Step 2: Classify {len(valid_files)} result files ---")
    summary_rows = []
    
    k100_time_spent = 0.0 

    for fp, fn, om1, om2, K, mode, sig_col in sorted(valid_files):
        file_start_time = time.time() 
        
        df = pd.read_csv(fp)
        df_sig = df[df[sig_col] == True].copy()
        
        if df_sig.empty:
            n_total, n_A = 0, 0
        else:
            f1_col = 'feature1' if 'feature1' in df.columns else 'feature_g'
            f2_col = 'feature2' if 'feature2' in df.columns else 'feature_h'
            
            df_cls = classify_ddc_pairs(df_sig, lmm_cache[om1], lmm_cache[om2], f1_col, f2_col)
            
            out_name = fn.replace(".csv", "_classified_target.csv")
            df_cls.to_csv(os.path.join(save_dir, out_name), index=False)
            
            n_total = len(df_cls)
            n_A = (df_cls['Category'] == "A_Subtle_Coordinated").sum()
        
        summary_rows.append({
            "Omics_Pair": f"{om1} & {om2}",
            "K": K,
            "Total_Significant": n_total,
            "Cat_A_Count": n_A,
            "Cat_A_Ratio": round(n_A / n_total, 3) if n_total > 0 else 0
        })
        
        file_end_time = time.time() 
        if K == target_k:
            k100_time_spent += (file_end_time - file_start_time)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values(by=['Omics_Pair', 'K'])
        
        print("\n=== Post-hoc Summary (Target Features Only) ===")
        print(df_summary)
        
        summary_path = os.path.join(save_dir, "final_posthoc_summary_target.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n✅ Task complete. Summary results saved to: {summary_path}")
        
        total_time = time.time() - global_start_time
        lmm_time = lmm_end_time - lmm_start_time
        print("\n" + "="*40)
        print("⏱️  Performance Time Report:")
        print(f"  • Total LMM core calculation time: {lmm_time:.2f} seconds")
        print(f"  • Classification and saving time only: {k100_time_spent:.2f} seconds")
        print(f"  • Total script execution time: {total_time:.2f} seconds")
        print("="*40 + "\n")