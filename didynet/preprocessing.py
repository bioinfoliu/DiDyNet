import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

def filter_by_label(df, name, ir_data):
    filtegray = df.merge(ir_data[['SubjectID', 'Time']], on=['SubjectID', 'Time'])
    print(f"{name} Count after filtering by label: {len(filtegray)}")
    return filtegray

def deduplicate_with_mean(df, name):
    key_cols = ['SubjectID', 'Time']
    value_cols = df.select_dtypes(include='number').columns.difference(key_cols)
    df_dedup = df.groupby(key_cols)[value_cols].mean().reset_index()
    print(f"{name} Deduplication complete: from {len(df)} -> {len(df_dedup)} rows")
    return df_dedup

def run_preprocessing(base_path):
    print("====================================")
    print("=== Step 1: Data Preprocessing and Sample Alignment ===")
    print("====================================")
    plot_path = os.path.join(base_path, "plot")
    output_path = os.path.join(base_path, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    cytokines_df = pd.read_csv(os.path.join(base_path, "data/raw_data/cytokines.csv"))
    proteomics_df = pd.read_csv(os.path.join(base_path, "data/raw_data/proteomics.csv"))
    transcriptome_df = pd.read_csv(os.path.join(base_path, "data/raw_data/RNA.csv"))

    print(f"Cytokines Raw Shape: {cytokines_df.shape}")
    print(f"Proteomics Raw Shape: {proteomics_df.shape}")
    print(f"Transcriptome Raw Shape: {transcriptome_df.shape}")

    IRIS_label = pd.read_csv(os.path.join(base_path, "data/raw_data/IRIS_label.csv"))
    ir_data = IRIS_label[IRIS_label['IRIS'].isin(['IS', 'IR'])]

    cytokines_filtered = filter_by_label(cytokines_df, "Cytokines", ir_data)
    proteomics_filtered = filter_by_label(proteomics_df, "Proteomics", ir_data)
    transcriptome_filtered = filter_by_label(transcriptome_df, "Transcriptome", ir_data)

    cytokines_clean = deduplicate_with_mean(cytokines_filtered, "Cytokines")
    proteomics_clean = deduplicate_with_mean(proteomics_filtered, "Proteomics")
    transcriptome_clean = deduplicate_with_mean(transcriptome_filtered, "Transcriptome")

    scaler = StandardScaler()
    cytokines_clean.iloc[:, 2:] = scaler.fit_transform(cytokines_clean.iloc[:, 2:])
    transcriptome_clean.iloc[:, 2:] = scaler.fit_transform(transcriptome_clean.iloc[:, 2:])
    proteomics_clean.iloc[:, 2:] = scaler.fit_transform(proteomics_clean.iloc[:, 2:])

    proteomics_clean.columns = proteomics_clean.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
    transcriptome_clean.columns = transcriptome_clean.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)

    print(f"Cytokines Remaining observations: {len(cytokines_clean)}, Shape: {cytokines_clean.shape}")
    print(f"Transcriptome Remaining observations: {len(transcriptome_clean)}, Shape: {transcriptome_clean.shape}")
    print(f"Proteomics Remaining observations: {len(proteomics_clean)}, Shape: {proteomics_clean.shape}")

    print(f"Cytokines Unique Subjects: {cytokines_clean['SubjectID'].nunique()}")
    print(f"Transcriptome Unique Subjects: {transcriptome_clean['SubjectID'].nunique()}")
    print(f"Proteomics Unique Subjects: {proteomics_clean['SubjectID'].nunique()}")

    shared_subjects = set(cytokines_clean['SubjectID']).intersection(
        set(transcriptome_clean['SubjectID'])
    ).intersection(
        set(proteomics_clean['SubjectID'])
    )

    print(f"\n=== Shared Sample Analysis ===")
    print(f"Shared Subjects across all three omics: {len(shared_subjects)}")

    cyto_shared_obs = len(cytokines_clean[cytokines_clean['SubjectID'].isin(shared_subjects)])
    trans_shared_obs = len(transcriptome_clean[transcriptome_clean['SubjectID'].isin(shared_subjects)])
    prot_shared_obs = len(proteomics_clean[proteomics_clean['SubjectID'].isin(shared_subjects)])

    print(f"Observations for shared subjects in Cytokines: {cyto_shared_obs}")
    print(f"Observations for shared subjects in Transcriptome: {trans_shared_obs}")
    print(f"Observations for shared subjects in Proteomics: {prot_shared_obs}\n")

    cytokines_clean = cytokines_clean[cytokines_clean['SubjectID'].isin(shared_subjects)].copy()
    proteomics_clean = proteomics_clean[proteomics_clean['SubjectID'].isin(shared_subjects)].copy()
    transcriptome_clean = transcriptome_clean[transcriptome_clean['SubjectID'].isin(shared_subjects)].copy()
    ir_data_clean = ir_data[ir_data['SubjectID'].isin(shared_subjects)].copy()

    print(f"=== Final Data Shape Confirmation (Shared Samples Only) ===")
    print(f"Cytokines Final Shape: {cytokines_clean.shape}")
    print(f"Transcriptome Final Shape: {transcriptome_clean.shape}")
    print(f"Proteomics Final Shape: {proteomics_clean.shape}")
    print(f"Label Final Shape: {ir_data_clean.shape}\n")

    save_path = os.path.join(base_path, "data/cleaned/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cytokines_clean.to_csv(os.path.join(save_path, "cytokines_cleaned.csv"), index=False)
    proteomics_clean.to_csv(os.path.join(save_path, "proteomics_cleaned.csv"), index=False)
    transcriptome_clean.to_csv(os.path.join(save_path, "transcriptome_cleaned.csv"), index=False)
    ir_data_clean.to_csv(os.path.join(save_path, "IRIS_label_cleaned.csv"), index=False)

    print(f"All cleaned files (shared samples only) saved to: {save_path}")
    return cytokines_clean, transcriptome_clean, proteomics_clean, ir_data_clean