from idextraction import Id_Graph, Node, Edge, List_of_Nodes, List_of_Edges, set_list_of_nodes
import matplotlib.pyplot as plt

if __name__ == "__main__":
    node_list = List_of_Nodes(node_list=[Node(variable_name="A", variable_type="chance", values=["a", "b"]),
                                         Node(variable_name="B", variable_type="chance", values=["c", "d"])])
    set_list_of_nodes(node_list)
    edge_list = List_of_Edges(edge_list=[Edge(
        condition="A", variable="B", probabilities={"a": {"c": 0.5, "d": 0.5}, "b": {"c": 0.3, "d": 0.7}})])
    id_graph = Id_Graph(node_list=node_list, edge_list=edge_list)

    node_list2 = List_of_Nodes(node_list=[Node(variable_name="A", variable_type="chance", values=["a", "e"]),
                                         Node(variable_name="B", variable_type="chance", values=["c", "d"])])
    set_list_of_nodes(node_list2)
    edge_list2 = List_of_Edges(edge_list=[Edge(
        condition="A", variable="B", probabilities={"a": {"c": 0.6, "d": 0.4},"e": {"c": 0.3, "d": 0.7}})])
    id_graph2 = Id_Graph(node_list=node_list2, edge_list=edge_list2)
    print(id_graph.compare(id_graph2))
    id_graph.draw()
    plt.show()

