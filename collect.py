from utils.utils import read_json, write_jsonl
import os

def compute_one_file(path):
    data = read_json(path)
    entro_list = []
    acc_list = []
    for idx in range(len(data)):
        sample = data[idx]
        entro_list.append(sample['Log_p']['token_entropy'])
        acc_list.append(sample['has_answer'])
    return sum(acc_list)/len(acc_list), sum(entro_list)/len(entro_list)

def compute_all_files(dir):
    choice_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    entro_list, acc_list, token_prob, ref_prob = [], [], [], []
    for item in paths:
        data = read_json(os.path.join(dir, item))
        temp_acc = []
        for idx in range(len(data)):
            sample = data[idx]
            # if sample['has_answer'] != 1:
            #     continue
            temp_acc.append(sample['has_answer'])
            entro_list.append(sample['Log_p']['token_entropy'])
            acc_list.append(sample['has_answer'])
            token_prob.append(sample['Log_p']['token probs'][choice_idx[sample['Res']]])
            ref_prob.append(sample['Log_p']['token probs'][choice_idx[sample['reference']]])
        print(sum(temp_acc) / len(temp_acc))
    print(f'count: {len(acc_list)}')
    print(f'avg acc； {sum(acc_list)/len(acc_list)}')
    print(f'avg entropy: {sum(entro_list)/len(entro_list)}')
    print(f'avg token prob: {sum(token_prob) / len(token_prob)}')
    print(f'avg ref prob: {sum(ref_prob) / len(ref_prob)}')
    return acc_list, token_prob

def compute_ece(acc_list, prob_list):
    acc_bin = [[] for _ in range(10)]
    prob_bin = [[] for _ in range(10)]
    for idx in range(len(prob_list)):
        bin_idx = int(prob_list[idx] * 10)
        acc_bin[bin_idx].append(acc_list[idx])
        prob_bin[bin_idx].append(prob_list[idx])
    for idx in range(len(acc_bin)):
        if len(acc_bin[idx]) == 0:
            continue
        print(f'bin {idx}')
        print(f'count: {len(acc_bin[idx])}')
        print(f'avg acc: {sum(acc_bin[idx])/len(acc_bin[idx])}')
        print(f'avg prob: {sum(prob_bin[idx])/len(prob_bin[idx])}')


acc, probs = compute_all_files('./res/mmlu/zero-shot')
compute_ece(acc, probs)