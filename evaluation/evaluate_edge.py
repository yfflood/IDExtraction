import sys
sys.path.append("")
from idextraction.graph import Id_Graph
from idextraction import InfluenceDiagram
import os
import json
import pickle
import pandas as pd

if __name__ == "__main__":
    node_path = "./data/node_adjusted"
    node_files = os.listdir(node_path)
    node_files.sort(key=lambda x: int(x.split(".")[0]))
    edge_true_path = "./data/edge"
    edge_files = os.listdir(edge_true_path)
    edge_files.sort(key=lambda x: int(x.split(".")[0]))
    edge_experiment_paths = [
        # "./experiments/edge/extraction/cot/edges_cot",
        "./experiments/edge/extraction/few_shot/edges_0shot",
        # "./experiments/edge/extraction/few_shot/edges_1shot",
        # "./experiments/edge/extraction/few_shot/edges_3shot",
        # "./experiments/edge/generation/cot/edges_cot",
        "./experiments/edge/generation/few_shot/edges_0shot",
        # "./experiments/edge/generation/few_shot/edges_1shot",
        # "./experiments/edge/generation/few_shot/edges_3shot"
    ]
    e_idx=0
    for e_idx in range(6):
        edge_experiment_path = edge_experiment_paths[e_idx]
        edge_predict_files = os.listdir(edge_experiment_path)
        try:
            edge_predict_files.remove("results.pkl")
            edge_predict_files.remove("ans.csv")
        except:
            pass
        edge_predict_files.sort(key=lambda x: int(x.split(".")[0]))
        results=[]
        for idx in range(len(node_files)):
            with open(f"{node_path}/{node_files[idx]}", "r", encoding="utf-8") as f:
                node_list = json.load(f)
            with open(f"{edge_true_path}/{edge_files[idx]}", "r", encoding="utf-8") as f:
                edge_true_list = json.load(f)
            with open(f"{edge_experiment_path}/{edge_predict_files[idx]}", "r", encoding="utf-8") as f:
                edge_predict_list = json.load(f)
            diagram = InfluenceDiagram(node_list, edge_true_list)
            graph_true = Id_Graph(diagram.to_base_nodes(), diagram.to_base_edges())
            try:
                diagram = InfluenceDiagram(node_list, edge_predict_list)
                graph_predict = Id_Graph(diagram.to_base_nodes(), diagram.to_base_edges())
                edit_distance=graph_predict.compare(graph_true)
            except:
                edit_distance=-1
            result={
                "idx":idx,
                "edit_distance":edit_distance
            }
            results.append(result)
            with open(f"{edge_experiment_path}/results.pkl", "wb") as f:
                pickle.dump(results, f)
        results_df=pd.DataFrame(results)
        results_df.to_csv(f"{edge_experiment_path}/ans.csv",index=False,encoding="utf-8-sig")
        print(f"{edge_experiment_path} finished")
    

            
    
