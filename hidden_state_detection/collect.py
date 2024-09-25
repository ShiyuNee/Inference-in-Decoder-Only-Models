import json
import torch
import os
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer
import pandas as pd
import random
random.seed(0)

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def arrange_hidden_states_for_single_layer(dir, total_layer):
    """
    得到每一层对应的hidden states和labels(labels对所有层相同)
    """
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    labels = [[] for _ in range(total_layer)]
    data = [[] for _ in range(total_layer)]
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            for layer in range(total_layer):
                labels[layer].append(sample['has_answer'])
                data[layer].append(sample['hidden_states'][layer])

    for layer in range(total_layer):
        torch.save(torch.tensor(data[layer]), f'./data/layer{layer}/data.pt')
        torch.save(torch.tensor(labels[layer]), f'./data/layer{layer}/labels.pt')

def arrange_data_for_all_layers(dir, task):
    """
    得到所有样本对应的hidden states
    """
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    print(paths)
    labels = []
    data = []
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            labels.append(sample['has_answer'])
            data.append(sample['hidden_states'])
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    print(f'data shape: {data.shape}')
    print(f'label shape: {labels.shape}')
    torch.save(data, f'./data/{task}/all_layers/data.pt')
    torch.save(labels, f'./data/{task}/all_layers/labels.pt')

def arrange_probs(dir):
    labels = []
    data = []
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            labels.append(sample['has_answer'])
            data.append(sample['output_states'])
        torch.save(torch.tensor(data), f'./data/zero-shot-output/data.pt')
        torch.save(torch.tensor(labels), f'./data/zero-shot-output/labels.pt')
        
def save_all_data():
    for layer in range(33):
        if not os.path.exists(f'./data/layer{layer}'):
            os.mkdir(f'./data/layer{layer}')
    arrange_hidden_states_for_single_layer('./data/zero-shot-hidden', 33)

def dev_acc(dir):
    """
    得到所有数据的acc(所有任务), ref_label的平均概率, pred_label的平均概率, 返回每个任务的acc
    """
    res = []
    choices = {'A':0, 'B':1, 'C':2, 'D':3}
    avg_prob = []
    ref_prob = []
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    acc = [[] for _ in range(len(paths))]
    for task_id in tqdm(range(len(paths))):
        sub_data = read_json(os.path.join(dir, paths[task_id]))
        for sample in sub_data:
            acc[task_id].append(sample['has_answer'])
            res.append(sample['has_answer'])
            avg_prob.append(max(sample['Log_p']['token probs']))
            print(f"token probs: {sample['Log_p']['token probs']}")
            if sample['has_answer'] == 0:
                if len(sample['Log_p']['token probs']) == 4:
                    ref_prob.append(sample['Log_p']['token probs'][choices[sample['reference']]])
                else:
                    ref_prob.append(max(sample['Log_p']['token probs'][choices[sample['reference']]], sample['Log_p']['token probs'][choices[sample['reference']]+4]))
    acc = [sum(item) / len(item) for item in acc]
    print(f'dev count: {len(res)}')
    print(f'ref prob: {sum(ref_prob) / len(ref_prob)}')
    print(f'avg prob: {sum(avg_prob) / len(avg_prob)}')
    print(f'acc: {sum(res) / len(res)}')
    return acc

def split_all_data_to_task_data(all_data, all_labels, dir):
    """
    将所有数据按task拆分
    Input:
        - all_data: 所有数据
        - dir: 包含所有task的文件夹路径
    Return:
        - task_data:[[]], 按task拆分后的数据
        - paths: [], 所有task的名称
    """
    total_count = 0
    tasks_count = []
    # 统计各个task的样本数量
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    for item in tqdm(paths):
        data = read_json(os.path.join(dir, item))
        tasks_count.append(len(data))
        total_count += len(data)
    # 将all_data分配到各个task内
    task_data = [[] for _ in range(len(tasks_count))]
    task_label = [[] for _ in range(len(tasks_count))]
    task_id = 0
    count = tasks_count[0]
    for idx in range(len(all_data)):
        task_data[task_id].append(all_data[idx])
        task_label[task_id].append(all_labels[idx])
        count -= 1
        if count == 0 and task_id < len(tasks_count) - 1:
            task_id += 1
            count = tasks_count[task_id]
    return task_data, task_label, paths

def get_out_token_count(path):
    data = read_json(path)
    token_cnt = []
    for item in data:
        token_cnt.append(len(item['Log_p']['tokens']))
    print(f'avg token count: {sum(token_cnt)/len(token_cnt)}')

def get_res_for_different_seed(base_dir):
    """
    统计不同seed训练得到的结果
    Example:
        base_dir = './data/nq/llama3-8b-instruct/mid_layer/zero-shot-chat/mid_layer/res/'
        get_res_for_different_seed(base_dir)
    """
    total_score = []
    for mode in ['first', 'last', 'avg']:
        mode_score = [0.0, 0.0]
        for seed in ['0', '42', '100']:
            if 'sample' in base_dir:
                file_path = base_dir + 'sample_' + mode + '_seed' + seed + '.jsonl'
            else:
                file_path = base_dir + mode + '_seed' + seed + '.jsonl'
            data = read_json(file_path)
            for idx in [0, 1]:
                mode_score[idx] += list(data[idx].values())[0]
        mode_score = [round(item / 3, 4) for item in mode_score]
        print(f'{mode}-avg score: {mode_score}')
        total_score.append(mode_score)

def compute_acc(path):
    data = read_json(path)
    res = []
    for item in data:
        res.append(item['has_answer'])
    print(f'count: {len(res)}')
    print(f'acc: {sum(res)/len(res)}')

def different_knowledge_level():
    qa_data = read_json('./data/nq/llama3-8b-instruct/mid_layer/zero-shot-chat/nq_test_llama8b_tokens_mid_layer.jsonl')
    mc_rand_data = read_json('./data/nq-mc/llama3-8b-instruct/mid_layer/zero-shot-gene/nq-test-gene-choice.jsonl')
    assert len(qa_data) == len(mc_rand_data)
    right2wrong = []
    wrong2right = []
    for idx in range(len(qa_data)):
        if qa_data[idx]['has_answer'] == 1 and mc_rand_data[idx]['has_answer'] == 0:
            right2wrong.append(1)
        if qa_data[idx]['has_answer'] == 0 and mc_rand_data[idx]['has_answer'] == 1:
            wrong2right.append(1)
        print(qa_data[idx]['Res'])
    print(f'right->wrong: {round(len(right2wrong)/len(qa_data), 4)}')
    print(f'wrong->right: {round(len(wrong2right)/len(qa_data), 4)}')

def sample_training_data_for_random_mc(rand_path, acc=0):
    wrong_list = []
    data = read_json(rand_path)
    for idx in range(len(data)):
        if data[idx]['has_answer'] == acc:
            wrong_list.append(idx)
    remain_idx = [item for item in range(len(data)) if item not in wrong_list]
    total_idx = wrong_list + random.sample(remain_idx, len(wrong_list))

    new_data = [data[idx] for idx in range(len(data)) if idx in total_idx]
    out_path = '/'.join(rand_path.split('/')[:-1]) + '/' + rand_path.split('/')[-1].replace('choice', 'choice-sample')
    write_jsonl(new_data, out_path)

def compute_acc_for_mc_task(ref_path, gene_path):
    """
    ref_path:数据集的path
    gene_path:跑出来结果的path
    """
    choices_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    ref_data = pd.read_csv(ref_path, header=None).to_numpy()
    gene_data = read_json(gene_path)
    acc = []
    assert len(ref_data) == len(gene_data)
    for idx in range(len(ref_data)):
        if gene_data[idx]['has_answer'] == 1 and ref_data[idx][choices_idx[ref_data[idx][-1]] + 1] != "None of the others":
            acc.append(1)
        else:
            acc.append(0)
    print(f'count: {len(acc)}')
    print(f'acc: {sum(acc)/len(acc)}')


if __name__ == '__main__':
    # model = 'llama3-8b-instruct'
    # chat_mode = 'zero-shot-none'
    # dataset = 'hq'
    # get_res_for_different_seed(f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/mid_layer/res/')
    # compute_acc(f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/{dataset}-test-gene-none.jsonl')

    # sample
    # for dataset in ['nq', 'hq']:
    #     for chat_mode in ['zero-shot-none']:
    #         train_sample_path = f'../share/res/{dataset}-mc/qwen2/mid_layer/{chat_mode}/{dataset}-train-none-choice.jsonl'
    #         sample_training_data_for_random_mc(train_sample_path, 1)

    ref_path = '../share/datasets/hq-mc/test/hq-test-gene-choice-without-gt-4_test.csv'
    gene_path= '../share/res/hq-mc/llama3-8b-instruct/mid_layer/zero-shot-wo-gt-4/hq-test-gene-choice-without-gt-4.jsonl'
    compute_acc_for_mc_task(ref_path, gene_path)



    


        






