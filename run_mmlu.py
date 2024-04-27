import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.prompt import get_prompt
from utils.data import QADataset, MMLUDataset
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
    parser.add_argument('--type', type=str, choices=['qa', 'qa_evidence', 'qa_cot', 'post_evidence_judge', 'prior', 'post', 'post_evidence', 'generate', 'new_prior', 'certain_prior', 'repeat', 'prior_evidence', 'prior_cot', 'paraphrase'], default='qa')
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')   
    parser.add_argument('--idx', type=str, default="")   
    parser.add_argument('--model_path', type=str, default="") 
    parser.add_argument('--batch_size', type=int, default=1)   
    parser.add_argument('--n_shot', type=int, default=-1)
    parser.add_argument('--task', type=str, default='mmlu')
    args = parser.parse_args()
    args.ra = ra_dict[args.ra]

    return args


def main():

    args = get_args()
    engine = Generater(args)
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.source, "test")) if "_test.csv" in f])
    accuracy = {}
    total_acc = 0
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)
    for subject in subjects:
        print(f'subject: {subject}')
        all_data = MMLUDataset(args, subject)
        engine.load_data(all_data)
        res, score = engine.get_res()
        accuracy[subject] = score
        total_acc += score
        write_jsonl(res, args.outfile + subject + '.jsonl')
    accuracy['total'] = total_acc / len(subjects)
    write_jsonl([accuracy], args.outfile + 'accuracy.jsonl')


if __name__ == '__main__':
    main()
