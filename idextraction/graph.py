import os
import json
import os
import sys 
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from idextraction.id import Node, Edge, List_of_Nodes, List_of_Edges, VariableType
import networkx as nx

import pycid
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def node_subst_cost(n1, n2):
    cost = 0
    if n1.get("variable_type") != n2.get("variable_type"):
        cost += 0.5
    intersection = len(set(n1.get("values")) & set(n2.get("values")))
    union = len(set(n1.get("values")) | set(n2.get("values")))
    cost += (1-(intersection/union))/2
    return cost


def edge_subst_cost(e1, e2):
    keys_condition = e1.get("probabilities").keys()
    keys_variable = list(e1.get("probabilities").values())[0].keys()
    costs = []
    for condition in keys_condition:
        for variable in keys_variable:
            costs.append(
                abs(
                    e1.get("probabilities").get(condition).get(variable)
                    - e2.get("probabilities").get(condition).get(variable)
                ) 
                if e2.get("probabilities").get(condition) is not None 
                and e2.get("probabilities").get(condition).get(variable) is not None
                else 1
            )
    cost = sum(costs)/len(costs)
    return cost


class Id_Graph:

    def __init__(self, node_list, edge_list):
        self.graph = nx.DiGraph()
        self.node_list: List_of_Nodes = node_list
        self.edge_list: List_of_Edges = edge_list
        self.graph.add_nodes_from(self.node_list.get_nodes())
        self.graph.add_edges_from(self.edge_list.get_edges())

    def draw(self):
        try:
            self.to_cid()
        except Exception as e:
            print(e)
            nx.draw(self.graph, with_labels=True)
            plt.show()

    def compare(self, other):
        return nx.graph_edit_distance(
            self.graph,
            other.graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost
        )


    def to_cid(self):  
        edges = self.edge_list.get_edges()
        edges = [(h, t) for h, t, _ in edges]

        nodes = self.node_list.get_nodes()
        decisions = [name for name, node in nodes if node["variable_type"]==VariableType.decision]
        utilities = [name for name, node in nodes if node["variable_type"]==VariableType.utility]

        cid = pycid.CID(
            edges, decisions=decisions, utilities=utilities
        )
        cid.draw()
        return cid

    def get_decision_nodes(self):
        nodes = self.node_list.get_nodes()
        decisions = [name for name, node in nodes if node["variable_type"]==VariableType.decision]
        return decisions

    def get_utility_nodes(self):
        nodes = self.node_list.get_nodes()
        utilities = [name for name, node in nodes if node["variable_type"]==VariableType.utility]
        return utilities
    
    def get_root_nodes(self):
        """ return a list of nodes which has no parent nodes"""
        nodes = self.node_list.get_nodes()
        edges = self.get_edges()
        non_root_nodes = list(set([t for h, t in edges]))
        root_nodes = [nodes.remove(node) for node in non_root_nodes]
        return root_nodes

    def get_edges(self):
        edges = self.edge_list.get_edges()
        edges = [(h, t) for h, t, _ in edges]
        return edges

    

if __name__=="__main__":
    node_files=os.listdir("./data/node_adjusted")
    edge_gen_files=os.listdir("./data/edge_generated")
    edge_ext_files=os.listdir("./data/edge_extracted")

    idx = 2
    print(node_files[idx])
    with open(f"./data/node_adjusted/{node_files[idx]}", "r", encoding="utf-8") as f:
        node_list = json.load(f)
    with open(f"./data/edge_generated/{edge_gen_files[idx]}", "r", encoding="utf-8") as f:
        edge_gen_list = json.load(f)
    with open(f"./data/edge_extracted/{edge_ext_files[idx]}", "r", encoding="utf-8") as f:
        edge_ext_list = json.load(f)
    
    list_of_node = List_of_Nodes([
        Node(
            node["variable_name"],
            node["variable_type"],
            node["values"]
        ) for node in node_list
    ])

    list_of_edge_gen = List_of_Edges([
        Edge(
            edge["condition"],
            edge["variable"],
            edge["probabilities"]
        ) for edge in edge_gen_list
    ])

    list_of_edge_ext = List_of_Edges([
        Edge(
            edge["condition"],
            edge["variable"],
            edge["probabilities"]
        ) for edge in edge_ext_list
    ])

    graph_gen = Id_Graph(list_of_node, list_of_edge_gen)
    graph_ext = Id_Graph(list_of_node, list_of_edge_ext)
    
    #graph_gen.to_cid()
    #print(list_of_edge_gen.get_edges()[0][0])
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
    #plt.figure(figsize=(8,4))
    #plt.subplot(121)
    plt.title("Generated")
    graph_gen.draw()

    #plt.subplot(122)
    plt.title("Extracted")
    graph_ext.draw()

    #plt.tight_layout()
    #plt.show()
    