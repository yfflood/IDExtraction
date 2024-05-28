from .id import Node, Edge, List_of_Nodes, List_of_Edges
import networkx as nx


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
                if e2.get("probabilities").get(condition).get(variable) is not None
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
        nx.draw(self.graph, with_labels=True)

    def compare(self, other):
        return nx.graph_edit_distance(
            self.graph,
            other.graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost
        )
