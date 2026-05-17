import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

def filter_by_label(df, name, ir_data):
    filtegray = df.merge(ir_data[['SubjectID', 'Time']], on=['SubjectID', 'Time'])
    print(f"{name} 筛选有标签样本后数量: {len(filtegray)}")
    return filtegray

def deduplicate_with_mean(df, name):
    key_cols = ['SubjectID', 'Time']
    value_cols = df.select_dtypes(include='number').columns.difference(key_cols)
    df_dedup = df.groupby(key_cols)[value_cols].mean().reset_index()
    print(f"{name} 去重完成：从 {len(df)} -> {len(df_dedup)} 行")
    return df_dedup

def run_preprocessing(base_path):
    print("====================================")
    print("=== Step 1: 数据预处理与样本对齐 ===")
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

    print(f"Cytokines 原始数据维度 (Raw Shape): {cytokines_df.shape}")
    print(f"Proteomics 原始数据维度 (Raw Shape): {proteomics_df.shape}")
    print(f"Transcriptome 原始数据维度 (Raw Shape): {transcriptome_df.shape}")

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

    print(f"Cytokines 剩余观测数量: {len(cytokines_clean)}，维度: {cytokines_clean.shape}")
    print(f"Transcriptome 剩余观测数量: {len(transcriptome_clean)}，维度: {transcriptome_clean.shape}")
    print(f"Proteomics 剩余观测数量: {len(proteomics_clean)}，维度: {proteomics_clean.shape}")

    print(f"Cytokines 独立样本数 (Unique Subjects): {cytokines_clean['SubjectID'].nunique()}")
    print(f"Transcriptome 独立样本数 (Unique Subjects): {transcriptome_clean['SubjectID'].nunique()}")
    print(f"Proteomics 独立样本数 (Unique Subjects): {proteomics_clean['SubjectID'].nunique()}")

    shared_subjects = set(cytokines_clean['SubjectID']).intersection(
        set(transcriptome_clean['SubjectID'])
    ).intersection(
        set(proteomics_clean['SubjectID'])
    )

    print(f"\n=== 交集样本分析 ===")
    print(f"三组学共有受试者数量 (Shared Subjects): {len(shared_subjects)}")

    cyto_shared_obs = len(cytokines_clean[cytokines_clean['SubjectID'].isin(shared_subjects)])
    trans_shared_obs = len(transcriptome_clean[transcriptome_clean['SubjectID'].isin(shared_subjects)])
    prot_shared_obs = len(proteomics_clean[proteomics_clean['SubjectID'].isin(shared_subjects)])

    print(f"交集样本在 Cytokines 中的观测数量: {cyto_shared_obs}")
    print(f"交集样本在 Transcriptome 中的观测数量: {trans_shared_obs}")
    print(f"交集样本在 Proteomics 中的观测数量: {prot_shared_obs}\n")

    cytokines_clean = cytokines_clean[cytokines_clean['SubjectID'].isin(shared_subjects)].copy()
    proteomics_clean = proteomics_clean[proteomics_clean['SubjectID'].isin(shared_subjects)].copy()
    transcriptome_clean = transcriptome_clean[transcriptome_clean['SubjectID'].isin(shared_subjects)].copy()
    ir_data_clean = ir_data[ir_data['SubjectID'].isin(shared_subjects)].copy()

    print(f"=== 最终落盘数据维度确认 (仅含共享样本) ===")
    print(f"Cytokines 最终保存维度: {cytokines_clean.shape}")
    print(f"Transcriptome 最终保存维度: {transcriptome_clean.shape}")
    print(f"Proteomics 最终保存维度: {proteomics_clean.shape}")
    print(f"Label 最终保存维度: {ir_data_clean.shape}\n")

    save_path = os.path.join(base_path, "data/cleaned/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cytokines_clean.to_csv(os.path.join(save_path, "cytokines_cleaned.csv"), index=False)
    proteomics_clean.to_csv(os.path.join(save_path, "proteomics_cleaned.csv"), index=False)
    transcriptome_clean.to_csv(os.path.join(save_path, "transcriptome_cleaned.csv"), index=False)
    ir_data_clean.to_csv(os.path.join(save_path, "IRIS_label_cleaned.csv"), index=False)

    print(f"所有清洗后的文件 (仅包含 shared samples) 已保存至: {save_path}")
    return cytokines_clean, transcriptome_clean, proteomics_clean, ir_data_clean
