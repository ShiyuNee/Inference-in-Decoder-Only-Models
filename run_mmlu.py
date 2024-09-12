import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.prompt import get_prompt
from utils.data import QADataset, MCDataset
from utils.llm import Generater
# from utils.llm_deepspeed import ParallelGenerater
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
    parser.add_argument('--type', type=str, choices=['mc_qa', 'mc_qa_evidence', 'mc_qa_cot'], default='qa')
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')   
    parser.add_argument('--idx', type=str, default="")   
    parser.add_argument('--model_path', type=str, default="") 
    parser.add_argument('--batch_size', type=int, default=1)   
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--task', type=str, default='mmlu')
    parser.add_argument('--with_answer', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
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
    # engine = ParallelGenerater(args)
    engine = Generater(args)
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.source, "test")) if "_test.csv" in f])
    print(f'subjects: {subjects}')
    accuracy = {}
    total_acc = 0
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    for idx in range(len(subjects)):
        subject = subjects[idx]
        if args.task == 'mmlu' or args.task == 'tq':
            all_data = MCDataset(args, subject)
            # 7200 for llama3-8b-instruct, 3000 for llama2-chat-7b
            if args.model_name == 'llama3-8b-instruct':
                engine.batch_size = int(7000 / (all_data.avg_len + args.max_new_tokens)) 
            elif args.model_name in ['llama2-7b-chat', 'qwen2-7b-instruct']:
                engine.batch_size = int(3000 / (all_data.avg_len + args.max_new_tokens)) # llama2运行时更耗显存
        else:
            raise ValueError(f'Specify the wrong task: {args.task}')
        print(f'cnt: {idx}, subject: {subject}, batch size: {engine.batch_size}')
        engine.load_data(all_data)
        res, score = engine.get_res()
        accuracy[subject] = score
        total_acc += score
        write_jsonl(res, args.outfile + subject + '.jsonl')
        
    accuracy['total'] = total_acc / len(subjects)
    write_jsonl([accuracy], args.outfile + 'accuracy.jsonl')


if __name__ == '__main__':
    main()
