import json

qids = set()
with open("plain/b489700156a128d5/eval_dev_out.json") as f:
    data = json.load(f)
    for data_item in data:
        qid = data_item["q_id"]
        qids.add(qid)

with open("plain/b489700156a128d5/processed_ielab_custom_baseline_blender_70b_filtered_meta-llama_Meta-Llama-3.1-70B-Instruct_llm_based_attribution_trec_rag_few_shots.jsonl") as f:
    for line in f:
        data_item = json.loads(line)
        qids.remove(data_item["topic_id"])


with open("plain/missed_qids.txt", "w") as f:
    for qid in qids:
        f.write(qid + "\n")