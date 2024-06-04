import os
import sys 
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from idextraction.id import *
from idextraction.similarity import phrase_distance_hf, is_synonymy_llm

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

import json
from tqdm import tqdm

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
            if is_synonymy_llm(v1, v2, chat_model)["synonymy"]:
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

        
def is_coreference(node1, node2, chat_model):
    """ 
    use chat_model to judge whether the 2 input nodes have coreference relation. nodes are considered synonymous if the following holds:
     - they have extremely similar variable_name
     - they have the same variable_type
     - the values they take can match

    node1, node2: type Node
    """
    # if chat model return synonymy
    if isinstance(node1, Dict):
        if is_synonymy_llm(node1["variable_name"], node2["variable_name"], chat_model)["synonymy"]==False:
            return False

        if node1["variable_type"] != node2["variable_type"]: 
            # TODO: 考虑加warning?
            return False

        return True

    elif isinstance(node1, Node):
        if is_synonymy_llm(node1.variable_name, node2["variable_name"], chat_model)["synonymy"]==False:
            return False

        if node1.variable_type != node2["variable_type"]: 
            # TODO
            return False

        return True


def aggregate_nodes_from_files(node_files, chat_model):
    """ aggregate the JSON files into a complete node list"""
    node_list = []
    # record the files each node appears
    # 
    source_dict = {} 
    for file in tqdm(node_files):
        with open(f"./data/node/{file}", "r", encoding="utf-8") as f:
            file_list=json.load(f)
            if len(file_list) == 0:
                continue
                
            # initialize the node_list
            if len(node_list) == 0:
                for node in file_list:
                    node_list.append(Node(node["variable_name"], node["variable_type"], node["values"]))
                
                for node in node_list:
                    source_dict[node.get_id()] = [file]

                continue
            
            # pairwise compare & aggregate nodes in node_list and file_list
            for new_node in file_list:
                for idx, node in enumerate(node_list):
                    if is_coreference(node, new_node, chat_model):
                        combined_node = combine_nodes(node, new_node, chat_model)
                        node_list[idx] = combined_node

                        source_dict[combined_node.get_id()] = source_dict.pop(node.get_id())
                        source_dict[combined_node.get_id()].append(file)

                        file_list.remove(new_node)
                        break
            
            for new_node in file_list:
                added_node = Node(new_node["variable_name"], new_node["variable_type"], new_node["values"])
                node_list.append(added_node)
                source_dict[added_node.get_id()] = [file]
    return node_list, source_dict



if __name__=='__main__':
    glm = ChatZhipuAI(
        temperature=0,
        model="glm-4",
        max_tokens=4096
    )
    #nodes = [
    #    {'variable_name': 'Fire_Spread',
    #     'variable_type': 'chance',
    #     'values': ['Rapid_Upward', 'Contained', #'Gradual_spread']},
    #    {'variable_name': 'Fire Situation',
    #     'variable_type': 'chance',
    #     'values': ['Under control', 'Out of control']}
    #]

    node_files=os.listdir("./data/node_adjusted")
    
    node_list, source_dict = aggregate_nodes_from_files(node_files, chat_model=glm)

    node_out_file="./outputs/nodes.json"
    source_dict_file="./outputs/nodes_source.json"
    with open(node_out_file,"w",encoding="utf-8") as f:
        json.dump(node_list, f,ensure_ascii=False, indent=2, default=node_to_json)
    with open(source_dict_file,"w",encoding="utf-8") as f:
        json.dump(source_dict,f,ensure_ascii=False, indent=2)