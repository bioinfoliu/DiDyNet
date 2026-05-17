import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import os

COLOR_PALETTE = {
    'cytokines': '#E64B35',    
    'proteomics': '#4DBBD5',   
    'transcriptome': '#00A087' 
}
DEFAULT_COLOR = '#B09C85'

def run_network_construction(base_path, input_edge_file, top_k_per_layer, ers_threshold):
    print("\n====================================")
    print("=== Step 5: Construct Robust Network and Plot ===")
    print("====================================")
    
    save_dir = os.path.join(base_path, "results/Robustness_All_SCC")
    os.makedirs(save_dir, exist_ok=True)
    
    output_fig_png = os.path.join(save_dir, "Figure_6_ForceDirected_Balanced_Core.png")
    output_fig_pdf = os.path.join(save_dir, "Figure_6_ForceDirected_Balanced_Core.pdf")

    edge_col = 'Edge'  
    weight_col = 'Abs_Avg_Diff' 
    fig_size = (12, 12) 

    print(f"📂 Reading summary file: {input_edge_file}")
    if not os.path.exists(input_edge_file):
        print(f"❌ File not found: {input_edge_file}! If you haven't run the robustness loop, please manually specify a valid SCC result file.")
        return
        
    df_edges = pd.read_csv(input_edge_file)

    if 'Frequency' in df_edges.columns:
        df_edges = df_edges[df_edges['Frequency'] >= ers_threshold]
        print(f"🛡️ Applied ERS >= {ers_threshold} filter.")
    
    print("🧮 Parsing Edge column and building highly robust consensus network...")
    G_full = nx.Graph()
    node_omics_map = {}
    
    for _, row in df_edges.iterrows():
        edge_str = row[edge_col]
        if ' -- ' not in edge_str:
            continue
            
        u_raw, v_raw = edge_str.split(' -- ')
        u_omics = u_raw.split(':')[0]
        v_omics = v_raw.split(':')[0]
        
        node_omics_map[u_raw] = u_omics
        node_omics_map[v_raw] = v_omics
        
        weight_val = row[weight_col] if weight_col in df_edges.columns else 1.0
        G_full.add_edge(u_raw, v_raw, weight=abs(weight_val))
            
    global_degrees = dict(G_full.degree())
    
    print(f"⚖️ Stratifying and extracting Top {top_k_per_layer} Hubs per layer...")
    selected_nodes = []
    
    for omics in ['cytokines', 'proteomics', 'transcriptome']:
        layer_nodes = [n for n in G_full.nodes() if node_omics_map.get(n) == omics]
        top_k = sorted(layer_nodes, key=lambda x: global_degrees[x], reverse=True)[:top_k_per_layer]
        selected_nodes.extend(top_k)
        print(f"   - Extracted {len(top_k)} Hub nodes for {omics} layer.")

    G_visual = G_full.subgraph(selected_nodes).copy()

    print("🔄 Calculating force-directed layout (Kamada-Kawai)...")
    pos = nx.kamada_kawai_layout(G_visual, scale=2.0)

    node_colors = []
    node_sizes = []
    labels = {}

    visual_degrees = [global_degrees[n] for n in G_visual.nodes()]
    if len(visual_degrees) == 0:
        print("❌ No nodes in the network, cannot plot.")
        return
        
    min_deg, max_deg = min(visual_degrees), max(visual_degrees)

    for node in G_visual.nodes():
        otype = node_omics_map.get(node, "unknown")
        node_colors.append(COLOR_PALETTE.get(otype, DEFAULT_COLOR))
        
        deg = global_degrees[node]
        norm_size = (deg - min_deg) / (max_deg - min_deg + 1e-5)
        node_sizes.append(300 + norm_size * 1200)
        
        labels[node] = node.split(':')[1] if ':' in node else node

    plt.figure(figsize=fig_size, dpi=300, facecolor='white')

    nx.draw_networkx_edges(G_visual, pos, width=1.0, alpha=0.15, edge_color='#95a5a6')
    nx.draw_networkx_nodes(G_visual, pos, node_color=node_colors, node_size=node_sizes, 
                           alpha=0.95, linewidths=2.0, edgecolors='white')

    text_items = nx.draw_networkx_labels(G_visual, pos, labels, font_size=9, font_weight='bold', 
                                         horizontalalignment='center', verticalalignment='center')

    for t in text_items.values():
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.8)])

    legend_handles = [mpatches.Patch(color=color, label=label.capitalize()) for label, color in COLOR_PALETTE.items()]
    plt.legend(handles=legend_handles, loc='lower right', fontsize=12, frameon=True, 
               title="Omics Modality", title_fontproperties={'weight':'bold', 'size':13}, bbox_to_anchor=(1, 0))

    plt.title("Topological Architecture of the Core Regulatory Network", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off') 

    plt.savefig(output_fig_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_fig_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✨ Plotting complete! Saved to:\n  - {output_fig_png}")