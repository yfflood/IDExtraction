import os

from model import Kimi

from generate_nodes_and_edges import get_node, extract_node, extract_edge_files, generate_edge_files
import json


def node_fs(chat_model):
    with open("./experiments/node/few_shot/prompts.json", 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    for item in prompts:
        n_shots = item["n_shots"]
        types = item["types"]
        prompt = item["prompt"]
        print((n_shots, types))
        dir_name = f"nodes_{n_shots}shot_{types}"

        if not os.path.exists(f"./experiments/node/few_shot/{dir_name}"):
            os.makedirs(f"./experiments/node/few_shot/{dir_name}")
        out_dir = f"./experiments/node/few_shot/{dir_name}"

        input_text_files=os.listdir("./data/node_adjusted")
        for file in input_text_files:
            get_node(file.replace(".json", ".txt"), chat_model, prompt_template=prompt, out_dir=out_dir)


def node_cot(chat_model):
    with open("./experiments/node/staged/prompts.json", 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    zs_cot_prompt = prompts[0]["prompt"]
    rule_cot_prompt = prompts[1]["prompt"]
    utility_prompt = prompts[2]["prompt"]
    decision_prompt = prompts[3]["prompt"]
    chance_prompt = prompts[4]["prompt"]
    
    input_text_files=os.listdir("./data/node_adjusted")
    
    if not os.path.exists("./experiments/node/staged/zs_cot"):
        os.makedirs("./experiments/node/staged/zs_cot")
    if not os.path.exists("./experiments/node/staged/rule_cot"):
        os.makedirs("./experiments/node/staged/rule_cot")
    if not os.path.exists("./experiments/node/staged/manual"):
        os.makedirs("./experiments/node/staged/manual")

    print("rule_CoT")
    out_dir = "./experiments/node/staged/rule_cot"
    for file in input_text_files:
        get_node(file.replace(".json", ".txt"), chat_model, prompt_template=rule_cot_prompt, out_dir=out_dir)

    print("zs_CoT")
    out_dir = "./experiments/node/staged/zs_cot"
    for file in input_text_files:
        get_node(file.replace(".json", ".txt"), chat_model, prompt_template=zs_cot_prompt, out_dir=out_dir)

    
    """ #TODO: solve node overlap
    print("manual")
    out_dir = "./experiments/node/staged/manual"
    for file in input_text_files:
        file_txt = file.replace(".json", ".txt")
        with open(f"./data/input_text/{file_txt}", "r", encoding="utf-8") as f:
            text=f.readlines()[0]
        filename = file.split(".")[0]

        utility_nodes=extract_node(text, chat_model, utility_prompt)
        decision_nodes=extract_node(text, chat_model, decision_prompt)
        chance_nodes=extract_node(text, chat_model, chance_prompt)

        nodes = utility_nodes + decision_nodes + chance_nodes
        nodes = [node for node in nodes if node["variable_type"] != ""]
        with open(f"{out_dir}/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False)
    """


def edge_fs_gen(chat_model):
    with open("./experiments/edge/generation/few_shot/prompts.json", 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    for item in prompts:
        n_shots = item["n_shots"]
        prompt = item["prompt"]
        print(n_shots)
        dir_name = f"edges_{n_shots}shot"

        if not os.path.exists(f"./experiments/edge/generation/few_shot/{dir_name}"):
            os.makedirs(f"./experiments/edge/generation/few_shot/{dir_name}")
        out_dir = f"./experiments/edge/generation/few_shot/{dir_name}"

        node_file_list=os.listdir("./data/node_adjusted")
        generate_edge_files(node_file_list, chat_model, prompt, out_dir=out_dir)

def edge_cot_gen(chat_model):
    with open("./experiments/edge/generation/cot/prompts.json", 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    for item in prompts:
        prompt = item["prompt"]
        print("cot")
        dir_name = f"edges_cot"

        if not os.path.exists(f"./experiments/edge/generation/cot/{dir_name}"):
            os.makedirs(f"./experiments/edge/generation/cot/{dir_name}")
        out_dir = f"./experiments/edge/generation/cot/{dir_name}"

        node_file_list=os.listdir("./data/node_adjusted")
        generate_edge_files(node_file_list, chat_model, prompt, out_dir=out_dir)


def edge_fs_ext(chat_model):
    with open("./experiments/edge/extraction/few_shot/prompts.json", 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    for item in prompts:
        n_shots = item["n_shots"]
        prompt = item["prompt"]
        print(n_shots)
        dir_name = f"edges_{n_shots}shot"

        if not os.path.exists(f"./experiments/edge/extraction/few_shot/{dir_name}"):
            os.makedirs(f"./experiments/edge/extraction/few_shot/{dir_name}")
        out_dir = f"./experiments/edge/extraction/few_shot/{dir_name}"

        node_file_list=os.listdir("./data/node_adjusted")
        extract_edge_files(node_file_list, chat_model, prompt, out_dir=out_dir)

def edge_cot_ext(chat_model):
    with open("./experiments/edge/extraction/cot/prompts.json", 'r', encoding="utf-8") as f:
        prompts = json.load(f)

    for item in prompts:
        prompt = item["prompt"]
        print("cot")
        dir_name = f"edges_cot"

        if not os.path.exists(f"./experiments/edge/extraction/cot/{dir_name}"):
            os.makedirs(f"./experiments/edge/extraction/cot/{dir_name}")
        out_dir = f"./experiments/edge/extraction/cot/{dir_name}"

        node_file_list=os.listdir("./data/node_adjusted")
        extract_edge_files(node_file_list, chat_model, prompt, out_dir=out_dir)


if __name__=="__main__":
    kimi = Kimi()
    #node_cot(kimi)
    #node_fs(kimi)
    edge_cot_ext(kimi)
    edge_fs_ext(kimi)
    edge_cot_gen(kimi)
    edge_fs_gen(kimi)