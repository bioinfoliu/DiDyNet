import argparse
import os
from didynet.preprocessing import run_preprocessing
from didynet.feature_selection import run_feature_selection
from didynet.dtw_analysis import run_dtw_pipeline
from didynet.post_hoc import run_post_hoc
from didynet.network import run_network_construction

def main():
    parser = argparse.ArgumentParser(description="DiDyNet Pipeline Entry Point")
    parser.add_argument('--base_path', type=str, default="", help="Base path for data and results")
    parser.add_argument('--k', type=int, default=100, help="Top K hyperparameter for 2D Variance Filtering")
    parser.add_argument('--top_hubs', type=int, default=20, help="Number of Top Hubs to extract per omics layer")
    parser.add_argument('--ers_threshold', type=float, default=0.90, help="Edge Reliability Score threshold")
    args = parser.parse_args()

    # Step 1: Preprocessing
    cyto_clean, trans_clean, prot_clean, ir_label = run_preprocessing(args.base_path)

    # Step 2: Feature Selection (2D Variance)
    cytokine_tops, transcriptome_tops, proteomics_tops = run_feature_selection(
        cyto_clean, trans_clean, prot_clean, args.base_path, args.k
    )

    # Step 3: DTW & Wilcoxon
    data_map = {
        "cytokines": cyto_clean,
        "transcriptome": trans_clean,
        "proteomics": prot_clean
    }
    results_map = {
        "cytokines": cytokine_tops,
        "transcriptome": transcriptome_tops,
        "proteomics": proteomics_tops
    }
    subjects_all = sorted(list(ir_label['SubjectID'].unique()))
    
    run_dtw_pipeline(data_map, results_map, ir_label, subjects_all, args.base_path, args.k)

    # Step 4: LMM Post-hoc
    run_post_hoc(args.base_path, args.k)

    # Step 5: Network Construction
    input_edge_file = os.path.join(args.base_path, "results/Robustness_All_SCC/Final_ALL_SCC_Edges_Robustness.csv")
    if os.path.exists(input_edge_file):
        run_network_construction(args.base_path, input_edge_file, args.top_hubs, args.ers_threshold)
    else:
        print(f"⚠️ Skipping network plotting: Consolidated network edge file not found ({input_edge_file}).")
        print("If you need to plot the network for a single run, please modify run_pipeline.py to point this path to your single post-hoc SCC result file.")

    print("\n🎉 DiDyNet core pipeline execution completed successfully!")

if __name__ == "__main__":
    main()