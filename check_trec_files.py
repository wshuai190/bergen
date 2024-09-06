import argparse
import json

def main():
    args = argparse.ArgumentParser()
    #trec file 1, trec file 2
    args.add_argument('--trec_file1', type=str, default="runs/custom/trec_biogen_2024.custom_baseline.pubmed-trec-biogen-2024.trec")
    args.add_argument('--trec_file2', type=str, default="")

    # top n
    args.add_argument('--top_n', type=int, default=20)

    #add original query file and output query file
    args.add_argument("--query_file", type=str, default="../trec_biogen_data/test/BioGen2024topics-json.txt")
    args.add_argument("--output_query_file", type=str, default="../trec_biogen_data/test_cut/BioGen2024topics-json.txt")

    args = args.parse_args()


    # see how many queries top n are different
    qid_to_docids1 = {}
    with open(args.trec_file1) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            if int(rank) > args.top_n:
                continue
            if qid not in qid_to_docids1:
                qid_to_docids1[qid] = []
            qid_to_docids1[qid].append(docid)

    qid_to_docids2 = {}
    with open(args.trec_file2) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            if int(rank) > args.top_n:
                continue
            if qid not in qid_to_docids2:
                qid_to_docids2[qid] = []
            qid_to_docids2[qid].append(docid)



    #then select the qids that top n are different without considering the order
    diff_qids = set()
    for qid in qid_to_docids1:
        if qid not in qid_to_docids2:
            diff_qids.add(qid)
        else:
            if set(qid_to_docids1[qid]) != set(qid_to_docids2[qid]):
                diff_qids.add(qid)

    print(f"Number of queries that top {args.top_n} are different: {len(diff_qids)}")

    #read the query file, and output the queries that are different
    if args.query_file:
        with open(args.query_file) as f:
            queries = json.load(f)["topics"]

        with open(args.output_query_file, "w") as f:
            output_dict = {"topics": []}
            for query in queries:
                if query["id"] in diff_qids:
                    output_dict["topics"].append(query)
            json.dump(output_dict, f, indent=4)







