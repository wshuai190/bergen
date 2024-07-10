from modules.metrics import em_single
from datasets import load_dataset
from matplotlib import pyplot as plt
import json 
from tqdm import tqdm
import os 
import numpy as np
import math

interval_num = 10
dataset = load_dataset("akariasai/PopQA")["test"]
dataset = dataset.rename_column("question", "content")
dataset = dataset.map(lambda example: {'label': eval(example['possible_answers'])})
dataset = dataset.remove_columns(["possible_answers","id","subj", "prop","obj","subj_id","prop_id",'obj_id','s_aliases','o_aliases','s_uri','o_uri','s_wiki_title','o_wiki_title'])
# generating the id, train_0 ... validation_0 validation_1
cid = ["test"+str(i) for i in range(len(dataset))]
ds = dataset.add_column("id", cid)


chunk_type = "qid" # chunk either by qid or absolute


if chunk_type=='absolute':
    # first get popularity of each qid using s_pop
    pop_dict = {}
    for item in ds:
        qid = item['id']
        pop_score = item["o_pop"]
        if pop_score not in pop_dict:
            pop_dict[pop_score] = []
        pop_dict[pop_score].append(qid)
    # sort the pop_dict
    pop_dict = dict(sorted(pop_dict.items()))

    # this file is for case study analysis of popqa, 
    popqa_result_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-7b-100w/result_other_datasets/popqa"
    folder_names = ["baseline_wo_retrieve", "baseline_w_retrieve", "COCOM-4", "COCOM-16", "COCOM-128", "COCOM-light-4", "COCOM-light-16", "COCOM-light-128"]
    file_name = "eval_dev_out.json"

    def read_file_and_eval(file_path):
        with open(file_path, 'r') as f:
            final_memtric_dict = {}
            # the current dict is actually a list, read through qid
            current_dict = json.load(f)
            for item in current_dict:
                qid = item['q_id']
                labels = item['label']
                response = item['response']
                metric = max(em_single(response, label) for label in labels)
                final_memtric_dict[qid] = metric
        return final_memtric_dict

    out_folder = "interval"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Initialize dictionaries to accumulate results for COCOM and COCOM-light
    results_cocom = {}
    results_cocom_light = {}

    for folder_name in tqdm(folder_names):
        current_file = popqa_result_folder + "/" + folder_name + "/" + file_name
        current_dict = read_file_and_eval(current_file)

        popularity_interval = {}
        min_score = min(pop_dict.keys())
        max_score = max(pop_dict.keys())
        for pop_score in pop_dict:
            current_interval = int((pop_score - min_score) * interval_num / (max_score - min_score))
            for qid in pop_dict[pop_score]:
                if qid in current_dict:
                    if current_interval not in popularity_interval:
                        popularity_interval[current_interval] = []
                    popularity_interval[current_interval].append(current_dict[qid])
        for interval in popularity_interval:
            popularity_interval[interval] = sum(popularity_interval[interval]) / len(popularity_interval[interval])

        # Store results in the appropriate dictionary
        if "COCOM-light" in folder_name:
            results_cocom_light[folder_name] = dict(sorted(popularity_interval.items()))
        elif "baseline" in folder_name:
            # Include baseline results in both COCOM and COCOM-light dictionaries
            results_cocom[folder_name] = dict(sorted(popularity_interval.items()))
            results_cocom_light[folder_name] = dict(sorted(popularity_interval.items()))
        else:
            results_cocom[folder_name] = dict(sorted(popularity_interval.items()))

    # Function to plot results
    def plot_results(results_dict, title, out_file):
        for setting, intervals in results_dict.items():
            plt.plot(intervals.keys(), intervals.values(), label=setting)
        plt.xlabel("Popularity Interval")
        plt.ylabel("EM")
        plt.ylim(0, 1)
        plt.legend()
        plt.title(title)
        plt.savefig(out_file)
        plt.close()

    # Generate plots for COCOM and COCOM-light
    plot_results(results_cocom, "COCOM Settings", os.path.join(out_folder, "cocom.png"))
    plot_results(results_cocom_light, "COCOM-light Settings", os.path.join(out_folder, "cocom_light.png"))
else:
    # first get popularity of each qid using s_pop
    pop_list = [(item['id'], item["o_pop"]) for item in ds]
    # sort the list by popularity score
    pop_list.sort(key=lambda x: x[1])

    # split the sorted list into intervals
    chunk_size = len(pop_list) // interval_num
    intervals = [pop_list[i:i + chunk_size] for i in range(0, len(pop_list), chunk_size)]
    # remove last if not enough
    if len(intervals) > interval_num:
        intervals.pop()

    # this file is for case study analysis of popqa, 
    popqa_result_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-7b-100w/result_other_datasets/popqa"
    folder_names = ["baseline_wo_retrieve", "baseline_w_retrieve", "COCOM-4", "COCOM-16", "COCOM-128", "COCOM-light-4", "COCOM-light-16", "COCOM-light-128"]
    file_name = "eval_dev_out.json"

    def read_file_and_eval(file_path):
        with open(file_path, 'r') as f:
            final_memtric_dict = {}
            # the current dict is actually a list, read through qid
            current_dict = json.load(f)
            for item in current_dict:
                qid = item['q_id']
                labels = item['label']
                response = item['response']
                metric = max(em_single(response, label) for label in labels)
                final_memtric_dict[qid] = metric
        return final_memtric_dict

    out_folder = "interval"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Initialize dictionaries to accumulate results for COCOM and COCOM-light
    results_cocom = {}
    results_cocom_light = {}

    for folder_name in tqdm(folder_names):
        current_file = popqa_result_folder + "/" + folder_name + "/" + file_name
        current_dict = read_file_and_eval(current_file)

        popularity_interval = {i: [] for i in range(len(intervals))}
        for i, interval in enumerate(intervals):
            print(i)
            for qid, _ in interval:
                if qid in current_dict:
                    popularity_interval[i].append(current_dict[qid])
        for interval in popularity_interval:
            if popularity_interval[interval]:
                popularity_interval[interval] = np.mean(popularity_interval[interval])
            else:
                popularity_interval[interval] = 0

        # Store results in the appropriate dictionary
        if "COCOM-light" in folder_name:
            results_cocom_light[folder_name] = dict(sorted(popularity_interval.items()))
       
        elif "COCOM" in folder_name:
            results_cocom[folder_name] = dict(sorted(popularity_interval.items()))
        # else:
        #     # Include baseline results in both COCOM and COCOM-light dictionaries
        #     results_cocom[folder_name] = dict(sorted(popularity_interval.items()))
        #     results_cocom_light[folder_name] = dict(sorted(popularity_interval.items()))

    # Function to plot results
    def plot_results(results_dict, title, out_file):
        for setting, intervals in results_dict.items():
            plt.plot(intervals.keys(), intervals.values(), label=setting)
        plt.xlabel("Popularity Interval")
        plt.ylabel("EM")
        plt.ylim(0, 1)
        plt.legend()
        plt.title(title)
        plt.savefig(out_file)
        plt.close()

    # Generate plots for COCOM and COCOM-light
    plot_results(results_cocom, "COCOM Settings", os.path.join(out_folder, "cocom.png"))
    plot_results(results_cocom_light, "COCOM-light Settings", os.path.join(out_folder, "cocom_light.png"))









