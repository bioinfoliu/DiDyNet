import os
import numpy as np
import pandas as pd

def compute_feature_variances(df, name):
    required = {'SubjectID', 'Time'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{name}: Missing required columns {missing}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ('SubjectID', 'Time')]

    if len(feature_cols) == 0:
        print(f"{name}: No numeric feature columns found, returning empty dataframe.")
        return pd.DataFrame(columns=['Feature','Var_Time_mean','Var_Subject_mean'])

    var_time_mean = df.groupby('Time')[feature_cols].var(ddof=1).mean()
    var_subject_mean = df.groupby('SubjectID')[feature_cols].var(ddof=1).mean()

    out = pd.DataFrame({
        'Feature': feature_cols,
        'Var_Time_mean': var_time_mean.values,
        'Var_Subject_mean': var_subject_mean.values
    })
    
    print(f"{name} Feature count (numeric): {len(out)}")
    return out

def select_and_save_by_variance(var_df, name, tops, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(tops, (int, float)):
        tops = [int(tops)]

    df = var_df.dropna(subset=['Var_Time_mean', 'Var_Subject_mean']).copy()
    if df.empty:
        print(f"{name}: Input var_df is empty (or all NaN), no top lists generated.")
        pd.DataFrame(columns=[
            "Dataset","K","Time_Threshold","Subject_Threshold",
            "N_Time_Top","N_Subject_Top","N_Intersection","N_Union"
        ]).to_csv(os.path.join(save_dir, f"{name}_thresholds_summary.csv"), index=False)
        return {}

    nfeat = len(df)
    summary_rows = []
    result = {}

    for k in tops:
        k = int(k)
        if k <= 0:
            continue

        kk = min(k, nfeat)
        if kk < k:
            print(f"{name}: K={k} exceeds feature count {nfeat}, truncated to {kk}.")

        time_thresh = df['Var_Time_mean'].nlargest(kk).min()
        subj_thresh = df['Var_Subject_mean'].nlargest(kk).min()

        time_top = df.nlargest(kk, 'Var_Time_mean')['Feature'].tolist()
        subj_top = df.nlargest(kk, 'Var_Subject_mean')['Feature'].tolist()
        inter = sorted(set(time_top).intersection(subj_top))
        union = sorted(set(time_top).union(subj_top))

        pd.Series(time_top, name="Feature").to_csv(os.path.join(save_dir, f"{name}_Time_top_{k}.csv"), index=False)
        pd.Series(subj_top, name="Feature").to_csv(os.path.join(save_dir, f"{name}_Subject_top_{k}.csv"), index=False)
        pd.Series(inter,    name="Feature").to_csv(os.path.join(save_dir, f"{name}_Intersect_top_{k}.csv"), index=False)
        pd.Series(union,    name="Feature").to_csv(os.path.join(save_dir, f"{name}_Union_top_{k}.csv"), index=False)

        summary_rows.append({
            "Dataset": name,
            "K": k,
            "Time_Threshold": float(time_thresh) if pd.notna(time_thresh) else None,
            "Subject_Threshold": float(subj_thresh) if pd.notna(subj_thresh) else None,
            "N_Time_Top": len(time_top),
            "N_Subject_Top": len(subj_top),
            "N_Intersection": len(inter),
            "N_Union": len(union)
        })

        result[k] = {
            "time_top": time_top,
            "subject_top": subj_top,
            "intersection": inter,
            "union": union,
            "time_thresh": float(time_thresh) if pd.notna(time_thresh) else None,
            "subject_thresh": float(subj_thresh) if pd.notna(subj_thresh) else None
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_dir, f"{name}_thresholds_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"{name}: Saved {len(summary_rows)*4} list CSVs and threshold summary -> {save_dir}")
    print(f"{name}: Threshold summary file -> {summary_path}")
    return result

def run_feature_selection(cytokines_clean, transcriptome_clean, proteomics_clean, base_path, target_k):
    print("\n====================================")
    print("=== Step 2: 2D Variance Feature Filtering ===")
    print("====================================")
    save_dir = os.path.join(base_path, "top_features")
    
    cytokine_vars = compute_feature_variances(cytokines_clean, "Cytokines")
    transcriptome_vars = compute_feature_variances(transcriptome_clean, "Transcriptome")
    proteomics_vars = compute_feature_variances(proteomics_clean, "Proteomics")

    tops_list = [target_k]
    cytokine_tops = select_and_save_by_variance(cytokine_vars, "Cytokines", tops=tops_list, save_dir=save_dir)
    transcriptome_tops = select_and_save_by_variance(transcriptome_vars, "Transcriptome", tops=tops_list, save_dir=save_dir)
    proteomics_tops = select_and_save_by_variance(proteomics_vars, "Proteomics", tops=tops_list, save_dir=save_dir)
    
    return cytokine_tops, transcriptome_tops, proteomics_tops