import os
import json
import sys 
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr, root_validator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import List, Dict
from enum import Enum
import uuid


class VariableType(Enum):
    decision = "decision"
    chance = "chance"
    utility = "utility"


list_of_nodes = None


def set_list_of_nodes(nodes):
    global list_of_nodes
    list_of_nodes = nodes


test = [{'variable_name': 'Fire_Separation_Measures',
         'variable_type': 'decision',
        'values': ['Implement', 'Not_Implement']},
        {'variable_name': 'Fire_Spread',
        'variable_type': 'chance',
         'values': ['Rapid_Upward', 'Contained']},
        {'variable_name': 'Fire_Integrity_Compromise',
        'variable_type': 'chance',
         'values': ['Compromised', 'Intact']},
        {'variable_name': 'Utility',
         'variable_type': 'utility',
         'values': ['Prevent_Rapid_Upward_Spread']}]


extract_template = """\
For the following text, identify the fundamental goal inplied by the context (which may not directly appear in the text), and extract every variable, action or event. Output using the following format:

text: {text}

{format_instructions}
"""


def node_to_json(node):
    return {
        "variable_name": node.variable_name,
        "variable_type":node.variable_type.value,
        "values": node.values,
        "__id": str(node.get_id())
    }

def node_to_json(node):
    return {
        "variable_name": node.variable_name,
        "variable_type":node.variable_type.value,
        "values": node.values,
        "__id": str(node.get_id())
    }

class Node(BaseModel):
    variable_name: str = Field(
        description="Extract the variable, entity, action or event related to decision-making, and output its name. Make sure the name is understandable.")
    variable_type: VariableType = Field(
        description="What is the nature of the variable? If it is an aleatory variable that cannot be intervened, output 'chance'; if it can be intervened and represents a decision to be made, output 'decision'; if it reflects the fundamental goal of decision-maker in the setting, output 'utility'.")
    values: list = Field(
        description="Extract the possible values of the variable, at least two values are needed. Complete with common knowledge if necessary, and output them as a Python list.")
    __id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid1)

    def __init__(self, variable_name, variable_type, values):
        super().__init__(variable_name=variable_name,
                         variable_type=variable_type, values=values)

    def __hash__(self):
        return self.__id

    def get_id(self):
        return str(self.__id)

    def set_id(self, id: str):
        self.__id = uuid.UUID(id)

class List_of_Nodes(BaseModel):
    node_list: List[Node]

    def __init__(self, node_list):
        super().__init__(node_list=node_list)

    def find(self, variable_name):
        for node in self.node_list:
            if node.variable_name == variable_name:
                return node
        return None

    def get_nodes(self):
        return [
            (
                node.variable_name,
                {
                    "variable_name": node.variable_name,
                    "variable_type": node.variable_type,
                    "values": node.values
                }
            )
            for node in self.node_list
        ]


class Edge(BaseModel):
    condition: str = Field(description="Extract the condition that influences the value of another variable, and output the name of the condition.")
    variable: str = Field(description="Extract the variable that is influenced by the condition, and output the name of the variable.")
    probabilities: Dict[str, Dict[str, str]] = Field(description="Extract the conditional probability distribution of the variable. For the mentioned value of the condition (condition_value), extract the mentioned probability (variable_value_probability) of the variable values (variable_value). The probability should be either a verbal description or a real number within the range from 0.0 to 1.0, where 1.0 means the variable takes the value almost surely. Be accurate about the expression. For example, when something tends to happen but not with absolute certainty, 'very likely' can be more accurate than '1.0'. Output as a dictionary, whose values are also dictionaries, in the form of {condition_value: {variable_value: variable_value_probability}}")
    __id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid1)
    def __init__(self, condition, variable, probabilities):
        super().__init__(condition=condition, variable=variable, probabilities=probabilities)

    def __hash__(self):
        return self.__id
    
    def get_id(self):
        return str(self.__id)

    def set_id(self, id: str):
        self.__id = uuid.UUID(id)

    #@root_validator()
    def check_probabilities(cls, info):
        probabilities = info.get("probabilities")
        node_condition = list_of_nodes.find(info.get("condition"))
        node_variable = list_of_nodes.find(info.get("variable"))
        for key, value in probabilities.items():
            if key not in node_condition.values:
                raise ValueError(
                    f"Invalid value {key} for variable {node_condition.variable_name}.")
            if sum(value.values()) != 1:
                raise ValueError(
                    f"Probabilities for variable {node_condition.variable_name} must sum to 1.")
            for key2, _ in value.items():
                if key2 not in node_variable.values:
                    raise ValueError(
                        f"Invalid value {key2} for variable {node_variable.variable_name}.")
        return info


class List_of_Edges(BaseModel):
    edge_list: List[Edge]

    def __init__(self, edge_list):
        super().__init__(edge_list=edge_list)

    def find(self, condition, variable):
        for edge in self.edge_list:
            if edge.condition == condition and edge.variable == variable:
                return edge
        return None

    def get_edges(self):
        return [
            (
                edge.condition,
                edge.variable,
                {
                    "condition": edge.condition,
                    "variable": edge.variable,
                    "probabilities": edge.probabilities
                }
            )
            for edge in self.edge_list
        ]


class InfluenceDiagram:
    def __init__(self, node_list: list, edge_list: list):
        """ 
        input: extracted list of Dict's
        
        """
        self.node_list = node_list
        self.edge_list = edge_list

    def get_edges(self):
        edges = [(edge["condition"], edge["variable"]) for edge in self.edge_list]
        return edges

    def get_nodes(self):
        nodes = [node["variable_name"] for node in self.node_list]
        return nodes
    
    def get_root_nodes(self):
        nodes = self.get_nodes()
        edges = self.get_edges()
        non_root_nodes = [t for h,t in edges]
        root_nodes = [node for node in nodes if node not in non_root_nodes]
        return root_nodes

    def to_base_nodes(self):
        """ turn node_list to BaseModel class instances"""
        list_of_node = List_of_Nodes([
            Node(
                node["variable_name"],
                node["variable_type"],
                node["values"]
            ) for node in self.node_list
        ])
        return list_of_node
    
    def to_base_edges(self):
        """ turn edge_list to BaseModel class instances"""
        list_of_edge = List_of_Edges([
            Edge(
                edge["condition"],
                edge["variable"],
                edge["probabilities"]
            ) for edge in self.edge_list
        ])
        return list_of_edge

if __name__ == "__main__":
    #parser = JsonOutputParser(pydantic_object=List_of_Nodes)
    #parser.get_format_instructions()
    #prompt = PromptTemplate(
    #    template=extract_template,
    #    input_variables=["text"],
    #    partial_variables={
    #        "format_instructions": parser.#get_format_instructions()},
    #)
    #print(prompt.format(text="The weather is sunny and warm."))
    #a = List_of_Nodes(node_list=test)
    #b = Edge(condition="Fire_Separation_Measures", #variable="Fire_Spread", probabilities={"Implement": {
    #        "Rapid_Upward": 0.8, "Contained": 0.2}, #"Not_Implement": {"Rapid_Upward": 0.2, "Contained": 0.8}})
    #a = Node(test[0]["variable_name"], test[0]["variable_type"], test[0]["values"])
    #print(node_to_json(node=a))
    from idextraction.graph import Id_Graph

    node_files=os.listdir("./data/node_adjusted")
    edge_gen_files=os.listdir("./data/edge_generated")
    
    idx = 1

    with open(f"./data/node_adjusted/{node_files[idx]}", "r", encoding="utf-8") as f:
        node_list = json.load(f)
    with open(f"./data/edge_generated/{edge_gen_files[idx]}", "r", encoding="utf-8") as f:
        edge_gen_list = json.load(f)

    diagram1 = InfluenceDiagram(node_list, edge_gen_list)

    graph1 = Id_Graph(diagram1.to_base_nodes(), diagram1.to_base_edges())

    graph1.draw()