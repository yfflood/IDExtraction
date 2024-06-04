import os
import sys 
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from idextraction.id import *
from idextraction.similarity import phrase_distance_hf, phrase_similarity_llm

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


def combine_values(node1, node2, chat_model):
    """
    combine the values of two input nodes
    return: List, values merged from original nodes
    """
    values1 = node1['values']
    values2 = node2['values']
    values_merged = []
    for v1 in values1:
        for v2 in values2:
            if phrase_similarity_llm(v1, v2, chat_model)["synonymy"]:
                values_merged.append(v1)
                values1.remove(v1)
                values2.remove(v2)
                break
    if len(values1) > 0:
        for v1 in values1:
            values_merged.append(v1)
    if len(values2) > 0:
        for v2 in values2:
            values_merged.append(v2)
    return values_merged


def combine_nodes(node1, node2, chat_model):
    """ 
    combine nodes that are **identified** to be identical
    - chat_model: if == None, use manual rules to combine the nodes
    """
    variable_name = node1["variable_name"]
    variable_type = node1["variable_type"]
    values = combine_values(node1, node2, chat_model)
    node_merged = Node(variable_name=variable_name, variable_type=variable_type, values=values)
    return node_merged

        
def is_coreference(node1, node2, chat_model=None):
    """ 
    use chat_model to judge whether the 2 input nodes have coreference relation. nodes are considered synonymous if the following holds:
     - they have extremely similar variable_name
     - they have the same variable_type
     - the values they take can match

    node1, node2: type Node
    """
    # if chat model return synonymy
    if phrase_similarity_llm(node1["variable_name"], node2["variable_name"], chat_model)["synonymy"]==False:
        return False

    if node1["variable_type"] != node2["variable_type"]: 
        # TODO: 考虑加warning?
        return False

    return True


if __name__=='__main__':
    llm = ChatZhipuAI(
        temperature=0,
        model="glm-4",
        max_tokens=4096
    )
    nodes = [
        {'variable_name': 'Fire_Spread',
         'variable_type': 'chance',
         'values': ['Rapid_Upward', 'Contained', 'Gradual_spread']},
        {'variable_name': 'Fire Situation',
         'variable_type': 'chance',
         'values': ['Under control', 'Out of control']}
    ]
    # new_node = combine_nodes(nodes[0],nodes[1],llm)
    #print(new_node)
    print(combine_nodes(nodes[0], nodes[1],llm))