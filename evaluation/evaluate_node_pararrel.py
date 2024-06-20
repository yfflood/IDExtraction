from idextraction import Id_Graph
import os
import json
from model import Kimi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import ast
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle


def if_same_compare(list1, list2):
    prompt = PromptTemplate(
        input_variables=["list1", "list2"],
        template="compare two lists: \n \
        [List1]: {list1} \n [List2]: {list2} \n \
        if an element A in [List1] is the **same meaning** to element B in [List2], \
        add a tuple (A, B) to the output list. \n \
        the format of the output list is like: \n \
        [(A1, B1), (A2, B2), ...] \n \
        ** Only output the output list ** \n"
    )
    parser = StrOutputParser()
    model = Kimi()
    chain = prompt | model | parser
    i = 0
    while True:
        i += 1
        ans = chain.invoke({"list1": list1, "list2": list2})
        try:
            if '[' in ans and ']' in ans:
                ans = ans[ans.index('['):ans.index(']')+1]
            compare_list = ast.literal_eval(ans)
        except:
            compare_list = []
        if len(compare_list) > 0 or i > 3:
            break
    return compare_list


def evaluation(idx, truth_path, experienment_path, node_truth_files, node_predict_files):
    # print(idx)
    with open(os.path.join(truth_path, node_truth_files[idx]), "r", encoding="utf-8") as f:
        node_truth_list = json.load(f)
    with open(os.path.join(experienment_path, node_predict_files[idx]), "r", encoding="utf-8") as f:
        node_predict_list = json.load(f)
    name_truth_list = [node.get("variable_name", "")
                       for node in node_truth_list]
    name_predict_list = [node.get("variable_name", "")
                         for node in node_predict_list]
    if os.path.exists(f"{experienment_path}/ans.csv"):
        results = pd.read_csv(f"{experienment_path}/ans.csv", encoding="utf-8-sig")
        name_compare_list = ast.literal_eval(results.loc[results["index"]==idx, "name_info"].values[0])
    else:
        name_compare_list = if_same_compare(name_truth_list, name_predict_list)
    name_recall = len(name_compare_list) / \
        len(node_truth_list) if len(node_truth_list) > 0 else 0
    name_precision = len(name_compare_list) / \
        len(node_predict_list) if len(node_predict_list) > 0 else 0
    if name_recall>1:
        name_recall=1
    if name_precision>1:
        name_precision=1
    # print(f"recall: {name_recall}")
    type_dict = {}
    value_dict = {}
    for pair in name_compare_list:
        true_name = pair[0]
        pred_name = pair[1]
        for true_node in node_truth_list:
            if true_node.get("variable_name", "") == true_name:
                for pred_node in node_predict_list:
                    if pred_node.get("variable_name", "") == pred_name:
                        if true_node.get("variable_type", "") == pred_node.get("variable_type", ""):
                            type_dict[true_name] = 1
                        else:
                            type_dict[true_name] = 0
                        if os.path.exists(f"{experienment_path}/ans.csv"):
                            results = pd.read_csv(f"{experienment_path}/ans.csv", encoding="utf-8-sig")
                            value_dict = ast.literal_eval(results.loc[results["index"]==idx, "value_info"].values[0])
                            value_compare_list=value_dict.get(true_name, {}).get("content", [])
                            value_dict={}
                        else:
                            value_compare_list = if_same_compare(
                                [str(true_name)+":"+str(i)
                                for i in true_node.get("values", [])],
                                [str(true_name)+":"+str(i)
                                for i in pred_node.get("values", [])]
                            )
                        value_dict[true_name] = {
                            "content": value_compare_list,
                            "recall": len(value_compare_list)/len(true_node.get("values", [])) if len(true_node.get("values", [])) > 0 else 0,
                            "precision": len(value_compare_list)/len(pred_node.get("values", [])) if len(pred_node.get("values", [])) > 0 else 0,
                        }
                        if value_dict[true_name]["recall"]>1:
                            value_dict[true_name]["recall"]=1
                        if value_dict[true_name]["precision"]>1:
                            value_dict[true_name]["precision"]=1
                        break
                if true_name in type_dict and true_name in value_dict:
                    continue
    type_accuracy = sum(type_dict.values()) / \
        len(type_dict) if len(type_dict) > 0 else 0
    if type_accuracy>1:
        type_accuracy=1
    # print(f"Type Accuracy: {type_accuracy}")
    value_recall = sum([value.get("recall", 0) for value in value_dict.values(
    )])/len(value_dict) if len(value_dict) > 0 else 0
    if value_recall>1:    
        value_recall=1
    valu_precision = sum([value.get("precision", 0) for value in value_dict.values(
    )])/len(value_dict) if len(value_dict) > 0 else 0
    if valu_precision>1:
        valu_precision=1
    # print(f"Value recall: {value_recall}")
    result = {
        "index": idx,
        "name_recall": name_recall,
        "name_precision": name_precision,
        "type_accuracy": type_accuracy,
        "value_recall": value_recall,
        "value_precision": valu_precision,
        "name_info": name_compare_list,
        "type_info": type_dict,
        "value_info": value_dict
    }
    return result


def parallel_evaluation(truth_path, experienment_path, node_truth_files, node_predict_files, max_workers=4):
    data = None
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluation, idx, truth_path,
                            experienment_path, node_truth_files, node_predict_files)
            for idx in range(len(node_truth_files))
        ]

        for future in tqdm(as_completed(futures), total=len(node_truth_files), position=0):
            result = future.result()
            result_df = pd.DataFrame([result])
            results.append(result_df)
            with open(f"{experienment_path}/results.pkl", "wb") as f:
                pickle.dump(results, f)

    if results:
        data = pd.concat(results, ignore_index=True)

    return data


if __name__ == "__main__":
    truth_path = "./data/node_adjusted"
    node_truth_files = os.listdir(truth_path)
    node_experiment_paths = [
        "./experiments/node/few_shot/nodes_0shot_0",
        "./experiments/node/few_shot/nodes_1shot_0",
        "./experiments/node/few_shot/nodes_3shot_0",
        "./experiments/node/few_shot/nodes_3shot_1",
        "./experiments/node/staged/rule_cot",
        "./experiments/node/staged/zs_cot",
    ]
    for e_idx in range(6):
        experienment_path = node_experiment_paths[e_idx]

        node_predict_files = os.listdir(experienment_path)
        try:
            node_predict_files.remove("results.pkl")
            node_predict_files.remove("ans.csv")
            node_predict_files.remove("ans.xlsx")
        except:
            pass
        node_truth_files.sort(key=lambda x: int(x.split(".")[0]))
        node_predict_files.sort(key=lambda x: int(x.split(".")[0]))

        # with open(f"{experienment_path}/results.pkl", "rb") as f:
        #     data = pickle.load(f)
        #     data=pd.concat(data, ignore_index=True)

        data = parallel_evaluation(
            truth_path, experienment_path, node_truth_files, node_predict_files, 16)
        data.to_csv(f"{experienment_path}/ans.csv",
                    index=False, encoding="utf-8-sig")
        data.to_excel(f"{experienment_path}/ans.xlsx",
                    index=False)