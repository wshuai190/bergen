from modules.metrics import em_single
from modules.metrics import match_single
import json
import numpy as np
from tqdm import tqdm

test_metric = "M" # could be M or EM
real_baseline = "COCOM-4" # this is where sig test compares to

result_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-7b-100w/result_other_datasets"
datasets= ["kilt_nq", "kilt_triviaqa", "kilt_hotpotqa", "asqa", "popqa"]

model_folders = ["autocompressor", "icae", "xrag_mistral", "xrag_mixtral", "baseline_w_retrieve", "baseline_wo_retrieve", "COCOM-light-4", "COCOM-light-16", "COCOM-light-128","COCOM-4", "COCOM-16", "COCOM-128"] 
out_file = "eval_dev_out.json"
average_metrics_file = "eval_dev_metrics.json"

# do significance test againest baseline_w_retrieve,

# read the file
def read_out_file(file_path, metric):
    with open(file_path, 'r') as f:
        final_dic = {}
        # the current dict is actally a list, read through qid
        current_dict = json.load(f)
        for item in current_dict:
            labels= item['label']
            response= item['response']
            if metric == "M":
                metric_for_qid = np.max([match_single(label, response) for label in labels])
            else:
                metric_for_qid = np.max([em_single(label, response) for label in labels])
            final_dic[item['q_id']] = metric_for_qid
    
    # sort the dictionary by qid
    final_dic = dict(sorted(final_dic.items()))
    result_list = list(final_dic.values())
    return result_list

def read_metrics_file(file_path, metric):
    with open(file_path, 'r') as f:
        return json.load(f)[metric]



def get_significance(results_one, results_two):
    #results one and two are lists of float values, return true if the difference is significant
    from scipy import stats
    sig_value = stats.ttest_rel(results_one, results_two)
    return sig_value.pvalue < 0.05




average_dict = {}
sig_dict = {}
for dataset in tqdm(datasets):
    baseline_average = read_metrics_file(f"{result_folder}/{dataset}/{real_baseline}/{average_metrics_file}", test_metric)
    baseline_list = read_out_file(f"{result_folder}/{dataset}/{real_baseline}/{out_file}", test_metric)

    for model_folder in tqdm(model_folders):
        model_list = read_out_file(f"{result_folder}/{dataset}/{model_folder}/{out_file}", test_metric)
        model_average = read_metrics_file(f"{result_folder}/{dataset}/{model_folder}/{average_metrics_file}", test_metric)
        sig = get_significance(baseline_list, model_list)
        if model_folder not in average_dict:
            average_dict[model_folder] = {}
            sig_dict[model_folder] = {}
        average_dict[model_folder][dataset] = model_average
        sig_dict[model_folder][dataset] = sig
    
print(sig_dict)
# print in latex format, for easy copy paste
for model_folder in model_folders:
    print(f"{model_folder} & ", end="")
    for dataset in datasets:
        # if significant, put a star
        if not sig_dict[model_folder][dataset]:
            print(f"{average_dict[model_folder][dataset]:.3f}* & ", end="")
        else:
            print(f"{average_dict[model_folder][dataset]:.3f} & ", end="")
    print("\\\\")



    



