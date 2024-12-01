import argparse
from collections import defaultdict
import json
from typing import Dict
import numpy as np
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval
from tqdm import tqdm

from src.tools.logging_tools import LOGGER

def read_trec_file(file: str) -> Dict[str, Dict[str, str]]:
    o = defaultdict(dict)
    
    with open(file, 'r') as f:
        for line in f:
            qid, _, pid, rank = line.strip().split()
            o[qid][pid] = int(rank)
    
    return o

def main(opts: argparse.Namespace):
    
    with open(opts.input_file, "r") as f:
        queries = [json.loads(line) for line in f]
        
    query_list = [query["output"]["revised_query"] for query in queries]
    qid_list = [query["id"] for query in queries]
    
    searcher = LuceneSearcher(opts.index_dir_path)
    searcher.set_bm25(opts.bm25_k1, opts.bm25_b)
    
    qrels = read_trec_file(opts.gold_file)
    
    hits = searcher.batch_search(query_list, qid_list, opts.num_hits, threads=opts.num_threads)
    runs = defaultdict(dict)
    
    for qid in tqdm(qid_list, total=len(qid_list), desc="Processing queries"):
        for i, item in enumerate(hits[qid]):
            score = item.score
            doc_id = item.docid[3:]
            runs[qid][str(doc_id)] = float(score)
    metrics = {"recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(runs)
    
    mrr_list = [v['recip_rank'] for v in results.values()]
    recall_100_list = [v['recall_100'] for v in results.values()]
    recall_20_list = [v['recall_20'] for v in results.values()]
    recall_10_list = [v['recall_10'] for v in results.values()]
    recall_5_list = [v['recall_5'] for v in results.values()]

    LOGGER.info(f"MRR: {np.mean(mrr_list)}")
    LOGGER.info(f"Recall@5: {np.mean(recall_5_list)}")
    LOGGER.info(f"Recall@10: {np.mean(recall_10_list)}")
    LOGGER.info(f"Recall@20: {np.mean(recall_20_list)}")
    LOGGER.info(f"Recall@100: {np.mean(recall_100_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--index_dir_path", type=str, required=True)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument("--num_hits", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=48)
    opts = parser.parse_args()
    main(opts)
    
    """
    pixi run python3 src/bm25_utils/bm25_topiocqa.py \
        --input_file outputs/eval/openai_eval/new_gpt-4o_temp=1.0_topiOCQA_dev.jsonl \
        --gold_file ./rsc/preprocessed/topiOCQA/dev_gold.trec \
        --index_dir_path ./outputs/indexes/bm25 \
        --bm25_k1 0.9 \
        --bm25_b 0.4 \
        --num_hits 100 \
        --num_threads 48
    """