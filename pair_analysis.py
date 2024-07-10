import json
import  random
baseline_norag_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-baseline/e3300804091e862c"
baseline_rag_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-baseline/32f9c2cd32ade5fa"
xrag_30_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-baseline/xrag_30"
icae_folder = "/nfs/data/calmar/dylan/naver-projects/bergen/icae/kilt_nq/c93452b7246e5253"
base_folder_cocom = "/nfs/data/calmar/dylan/naver-projects/bergen/mistral-7b-combined-qa"
compression_rates = [4, 16, 128]
BERT_folders = ["5edc857ce081d885", "c00b1e47b8374cb4", "7f22f5d6f216d16b"]
DECODER_folders = ["cd96072726486cbb", "799cc7eb72086e42", "406fc632308cf080"]

file_name = "eval_dev_out.json"

def read_file(file_path):
    with open(file_path, 'r') as f:
        final_dic = {}
        # the current dict is actally a list, read through qid
        current_dict = json.load(f)
        for item in current_dict:
            final_dic[item['q_id']] = item


    return final_dic

baseline_no_dict = read_file(f"{baseline_norag_folder}/{file_name}")
baseline_dict = read_file(f"{baseline_rag_folder}/{file_name}")
xrag_30_dict = read_file(f"{xrag_30_folder}/{file_name}")
icae_dict = read_file(f"{icae_folder}/{file_name}")
bert_final_dict = {}

for (rate, bert_folder) in zip(compression_rates, BERT_folders):
    bert_dict = read_file(f"{base_folder_cocom}/{bert_folder}/{file_name}")
    for qid in bert_dict:
        if qid not in bert_final_dict:
            bert_final_dict[qid] = {}
        bert_final_dict[qid][rate] = bert_dict[qid]

decoder_final_dict = {}
for (rate, decoder_folder) in zip(compression_rates, DECODER_folders):
    decoder_dict = read_file(f"{base_folder_cocom}/{decoder_folder}/{file_name}")
    for qid in decoder_dict:
        if qid not in decoder_final_dict:
            decoder_final_dict[qid] = {}
        decoder_final_dict[qid][rate] = decoder_dict[qid]

while True:
    qid = input("Enter qid: ")
    if qid == "exit":
        break
    if qid == "random":
        qid = random.choice(list(baseline_dict.keys()))
    print(f"QID: {qid}")
    print("Instruction")
    print(baseline_dict[qid]["instruction"])
    print("Labels")
    print(baseline_dict[qid]["label"])
    
    print("-"*20)
    print("Baseline NoRAG Response")
    print(baseline_no_dict[qid]["response"])
    print("-"*20)
    print("Baseline RAGResponse")
    print(baseline_dict[qid]["response"])
    print("-"*20)
    print("Xrag 30 Response")
    print(xrag_30_dict[qid]["response"])
    print("-"*20)
    print("ICAE Response")
    print(icae_dict[qid]["response"])
    print("-"*20)
    for rate in compression_rates:
        print(f"Compression Rate: {rate}")
        print(f"COCOM:", decoder_final_dict[qid][rate]["response"])
        print(f"COCOM-light:",bert_final_dict[qid][rate]["response"])
        print("-"*20)

            
        
        
