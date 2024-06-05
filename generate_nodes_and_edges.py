import os
import langchain_core
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import RetryOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from idextraction import List_of_Nodes, List_of_Edges

from model import Kimi
from langchain_community.chat_models import ChatZhipuAI

import json
import time

os.environ["ZHIPUAI_API_KEY"] = '8adac289dfd59cb2006ccea31a82efab.GhrrpvmaygKVqjxS'


def extract_node(text, chat_model):
    extract_template = """\
    For the following text, extract every variable, entity, action or event. If none is found, output a default node, with name and type filled with empty strings and values an empty list.

    text: {text}

    Output using the following format:
    {format_instructions}

    The content should be in Chinese if the text is in Chinese.
    """
    parser = JsonOutputParser(pydantic_object=List_of_Nodes)

    prompt = PromptTemplate(
        template=extract_template,
        input_variables=["text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_model | parser
    contents = chain.invoke({"text": text})
    return contents['node_list']

def get_node(file, chat_model):
    """ extract nodes from file"""
    with open(f"./data/input_text/{file}", "r", encoding="utf-8") as f:
        text=f.readlines()[0]
    nodes=extract_node(text)
    filename=file.replace(".txt","")
    print(filename)
    with open(f"./data/node/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False)


def extract_edge(text, node_list, chat_model):
    extract_template = """\
    For the following list of nodes, identify influence relations between the variables based on the text, and assign to each influential relation a corresponding conditional or unconditional probability. All conditions and variables should be members of the list: {node_list}. 
    
    text: {text}

    Output using the following format:
    {format_instructions}

    The content should be in Chinese if the text is in Chinese.
    """
    parser = JsonOutputParser(pydantic_object=List_of_Edges)
    
    prompt = PromptTemplate(
        template=extract_template,
        input_variables=["text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_list": node_list
        }
    )
    
    try:
        chain = prompt | chat_model | parser
        contents = chain.invoke({"text": text})
    except langchain_core.exceptions.OutputParserException:
        print("parsing failed.")

        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=ChatZhipuAI(temperature=0, model="glm-4", max_tokens=4096), max_retries=10)#ChatZhipuAI(temperature=0, model="glm-4", max_tokens=4096)

        completion_chain = prompt | chat_model
        main_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))
        contents = main_chain.invoke({"text": text})
    return contents['edge_list']


def generate_edge(node_list, chat_model):
    """ use commonsense knowledge in the llm to directly generate edges""" 
    generate_template = """\
    For the following list of nodes, identify influence relations between the variables, and assign to each influential relation a corresponding conditional or unconditional probability. All conditions and variables should be members of the list: {node_list}. 

    Output using the following format:
    {format_instructions}

    The content should be in Chinese if the text is in Chinese.
    """
    parser = JsonOutputParser(pydantic_object=List_of_Edges)
    
    prompt = PromptTemplate(
        template = generate_template,
        input_variables=["node_list"],
        partial_variables={
            "format_instructions":parser.get_format_instructions(),
        }
    )
    try:
        chain = prompt | chat_model | parser
        contents = chain.invoke({"node_list": node_list})
    except langchain_core.exceptions.OutputParserException:
        print("kimi parsing failed.")
        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=ChatZhipuAI(temperature=0, model="glm-4", max_tokens=4096))

        completion_chain = prompt | chat_model
        main_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))
        contents = main_chain.invoke({"node_list": node_list})

    return contents['edge_list']


def generate_edge_files(node_file_list, chat_model):
    for node_file in node_file_list:
        filename = node_file.split(".")[0]
        print(filename)
        with open(f"./data/node_adjusted/{node_file}", "r", encoding="utf-8") as f:
            node_list=json.load(f)
        edge_list = generate_edge(node_list, chat_model)

        with open(f"./data/edge_generated/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(edge_list, f, ensure_ascii=False, indent=4)
        

def extract_edge_files(node_file_list, chat_model):
    for node_file in node_file_list:
        filename = node_file.split(".")[0]
        print(filename)
        with open(f"./data/node_adjusted/{node_file}", "r", encoding="utf-8") as f:
            node_list=json.load(f)
        with open(f"./data/input_text/{filename}.txt", "r", encoding="utf-8") as f:
            text=f.readlines()[0]

        edge_list = extract_edge(text, node_list, chat_model)

        with open(f"./data/edge_extracted/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(edge_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #if not os.path.exists("./data/node"):
    #    os.makedirs("./data/node")
    #if not os.path.exists("./data/edge"):
    #    os.makedirs("./data/edge")
    #input_text_files=os.listdir("./data/input_text")
    #for file in input_text_files:
    #    try:
    #        get_node(file)
    #    except Exception as e:
    #        time.sleep(5)
    #        get_node(file)
    kimi=Kimi()
    glm = ChatZhipuAI(
        temperature=0,
        model="glm-4",
        max_tokens=4096
    )
    text = """Openings connecting upper and
    lower floors within buildings compromise the in-
    tegrity of fire compartments, potentially leading to
    the spread of fire across multiple areas and floors.
    Therefore, reliable fire separation measures should
    be implemented in these connected spaces to pre-
    vent the rapid upward spread of fire."""
    #node_list = extract_node(text, glm)
    #print(node_list)
    #edge_list = generate_edge(node_list, glm)
    #print(edge_list)

    node_files=os.listdir("./data/node_adjusted")
    # node_files=["373.json"]
    #generate_edge_files(node_files, kimi)
    extract_edge_files(node_files, kimi)