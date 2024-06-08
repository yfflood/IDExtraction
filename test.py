from idextraction import Id_Graph, InfluenceDiagram
from generate_nodes_and_edges import extract_node, extract_edge
import matplotlib.pyplot as plt
from model import Kimi
import json
from tqdm import tqdm

if __name__ == "__main__":
    kimi = Kimi()
    with open("./experiments/node/few_shot/prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    with open("./experiments/edge/extraction/few_shot/prompts.json", "r", encoding="utf-8") as f:
        edge_prompts = json.load(f)
    with open("./case_study/case.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
    # node_lists=[]
    # for text in tqdm(texts):
    #     node_list = extract_node(
    #         text, kimi, extract_template=prompts[3]["prompt"])
    #     node_lists+=node_list
    # with open("./case_study/node.json", "w", encoding="utf-8") as f:
    #     json.dump(node_lists, f, ensure_ascii=False, indent=4)
    with open("./case_study/node.json", "r", encoding="utf-8") as f:
        node_lists = json.load(f)
    # edge_lists = []
    # for text in tqdm(texts):
    #     edge_list = extract_edge(text=text, chat_model=kimi, node_list=node_lists, extract_template=edge_prompts[1]["prompt"])
    #     # print(edge_list)
    #     edge_lists += edge_list
    # with open("./case_study/edge.json", "w", encoding="utf-8") as f:
    #     json.dump(edge_lists, f, ensure_ascii=False, indent=4)
    with open("./case_study/edge.json", "r", encoding="utf-8") as f:
        edge_lists = json.load(f)
    diagram1 = InfluenceDiagram(node_lists, edge_lists)

    graph1 = Id_Graph(diagram1.to_base_nodes(), diagram1.to_base_edges())

    graph1.draw()
