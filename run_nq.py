import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.prompt import get_prompt
from utils.data import QADataset
from utils.llm import Generater
from utils.utils import write_jsonl


ra_dict = {
    'none': 'none',
    'sparse': {'sparse_ctxs': 1},
    'dense': {'dense_ctxs': 1},
    'chatgpt': {'gen_ctxs': 100},
    'sparse+dense': {'dense_ctxs': 5, 'sparse_ctxs': 5},
    'gold': {'gold_ctxs': 1},
    'strong': {'strong_ctxs': 10},
    'weak': {'weak_ctxs': 10},
    'rand': {'rand_ctxs': 10},
    'dpr': {'dpr_ctx': 1},
    'extract': {'dpr_ctx': 1},
    'dpr_wrong': {'dpr_ctx_wrong': 1}
}


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/source/nq.json')
    parser.add_argument('--response', type=str, default='')
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--type', type=str, choices=['qa', 'qa_evidence', 'qa_cot', 'qa_more', 'qa_extract', 'qa_prior'], default='qa')
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')   
    parser.add_argument('--idx', type=str, default="")   
    parser.add_argument('--model_path', type=str, default="") 
    parser.add_argument('--batch_size', type=int, default=1)   
    parser.add_argument('--task', type=str, default='nq')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--hidden_states', type=bool, default=False)
    parser.add_argument('--output_states', type=bool, default=False)
    parser.add_argument('--attn_weights', type=bool, default=False)
    parser.add_argument('--hidden_idx_mode', type=str, default='last')
    parser.add_argument('--need_layers', type=str, default='last', choices=['all', 'last', 'mid'])
    args = parser.parse_args()
    args.ra = ra_dict[args.ra]
    args.model_name = args.model_path.split('/')[-1].replace('_', '-').lower()

    return args


def main():

    args = get_args()
    print(args)
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')

    all_data = QADataset(args)
    engine = Generater(args)
    engine.load_data(all_data)
    res, score = engine.get_res()
    write_jsonl(res, args.outfile)


if __name__ == '__main__':
    main()
