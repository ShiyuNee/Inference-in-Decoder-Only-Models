from utils.utils import read_json, write_jsonl
import os
import pandas as pd
import torch

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
    print(f'avg acc: {sum(acc_list)/len(acc_list)}')
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

def get_align_for_verbalized_conf(acc_path, verb_conf):
    answer_data = read_json(acc_path)
    conf_data = read_json(verb_conf)
    align = []
    assert len(answer_data) == len(conf_data)
    for idx in range(len(answer_data)):
        if answer_data[idx]['has_answer'] != conf_data[idx]['has_answer']:
            align.append(1)
        else:
            align.append(0)
    print(f'count: {len(align)}')
    print(f'align: {sum(align)/len(align)}')

def get_align_for_verbalized_conf_for_dir(acc_dir, verb_dir):
    acc_list = []
    conf_list = []
    align = []
    acc_paths = sorted([f for f in os.listdir(acc_dir) if ".jsonl" in f])
    for item in acc_paths:
        acc_data = read_json(os.path.join(acc_dir, item))
        conf_data = read_json(os.path.join(verb_dir, item))
        for idx in range(len(acc_data)):
            acc_list.append(acc_data[idx]['has_answer'])
            conf_list.append(conf_data[idx]['has_answer'])
            if acc_list[-1] != conf_list[-1]:
                align.append(1)
            else:
                align.append(0)
    print(f'count: {len(acc_list)}')
    print(f'acc: {sum(acc_list)/len(acc_list)}')
    print(f'uncertain: {sum(conf_list)/len(conf_list)}')
    print(f'align: {sum(align)/len(align)}')

def consistency_for_different_fils(question_path_list, freeform_path, ans_conf_path_list):
    """
    是否都选择了freeform中生成的选项
    - data_path:构造的各种选择题文件
    - ans_conf_path_list:第一个元素是freeform文件的path,后面是构造成各种选择题的文件的path
    还没写完
    """
    # freeform的在第一个
    choice2answer = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
    data = []
    question = []
    # 先找到freeform ans是哪个选项
    freeform_data = read_json(freeform_path)
    freeform_ans = [item['Res'] for item in freeform_data]
    choice_list = []

    for path in question_path_list:
        question.append(pd.read_csv(path), header=None).to_numpy()
        temp_choice = []
        for idx in range(len(question)):
            choice_idx = question[idx].index(freeform_ans[idx])
            temp_choice.append(choice2answer[choice_idx-1])
        choice_list.append(temp_choice)
                
    # 有没有选这个选项
    for path in ans_conf_path_list:
        data.append(read_json(path))

def alignment_match_for_different_files(freeform_conf_path, ans_conf_path_list, conf_label_path):
    """
    1.聚合多个问题的confidence并返回二维列表,其中第一个元素是freeform下的conf_list
    2.分析不同难度选择题与freeform下conf的consistent程度
    Input
        - freeform_align_path: freeform下对测试集的conf_path
        - ans_conf_path_list: 多种难度的选择题形式下的conf_list组成的二维列表
        - conf_label_path: freefrom下测试集的真实conf_label_path
    Return
        - ans_conf: 所有形式conf的二维列表,第一个元素为freeform对应的列表
        - gt_conf: freeform下的测试集ground truth conf列表
    """
    freeform_conf = read_json(freeform_conf_path)[0]['test_pred']
    gt_conf = torch.load(conf_label_path).tolist()
    ans_conf = []
    align_match = []

    for path in ans_conf_path_list:
        consis_align = []
        not_consis_align = []
        temp_conf = read_json(path)[0]['test_pred']
        ans_conf.append(temp_conf)
        temp_match = []
        for idx in range(len(temp_conf)):
            if temp_conf[idx] == freeform_conf[idx]:
                temp_match.append(1)
                consis_align.append(freeform_conf[idx] == gt_conf[idx])
            else:
                temp_match.append(0)
                not_consis_align.append(freeform_conf[idx] == gt_conf[idx])
        # print(f'consis align: {sum(consis_align)/len(consis_align)}')
        # print(f'not consis align: {sum(not_consis_align)/len(not_consis_align)}')
        align_match.append(temp_match)
    align_match = [sum(item)/len(item) for item in align_match]
    ans_conf.insert(0, freeform_conf)
    return ans_conf, gt_conf

def majority_vote(lists):
    """
    大多数投票算法
    """
    # 假设所有列表长度相同，获取第一个列表的长度
    list_length = len(lists[0])
    
    # 保存投票结果
    result = []
    align = []
    
    # 对每个位置进行大多数投票
    for i in range(list_length):
        # 统计当前位置上的 0 和 1 的数量
        vote_count = sum([lst[i] for lst in lists])
        
        # 如果超过一半是1，就将该位置的值设为1，否则设为0
        if vote_count > len(lists) / 2:
            result.append(1)
        else:
            result.append(0)
    return result

def rule_based_cooperate(conf_lists, gt_conf):
    """
    基于规则的,在多个conf_list间相互配合
    Input:
        - conf_lists: 多个难度的问题下,对结果的confidence list
        - gt_conf: 测试集freeform真实accuracy
    """
    consis_list = []
    not_consis_list = []
    for i in range(len(conf_lists[0])):
        vote_num = sum([lst[i] for lst in conf_lists])
        if vote_num == len(conf_lists) or vote_num == 0:
            consis_list.append(conf_lists[0][i] == gt_conf[i])
        else:
            # 这个规则不能加,加了效果会变差
            # if all_list[0][i] == 0 and vote_num >=4:
            #     all_list[0][i] = 1
            # 若freeform时判断能做对,但其余形式都做不对,则freeform判断可能错误
            if conf_lists[0][i] == 1 and vote_num <=1:
                conf_lists[0][i] = 0
            not_consis_list.append(conf_lists[0][i] == gt_conf[i])
    total_align = consis_list + not_consis_list
    return sum(total_align)/len(total_align)

def rule_based_alignment_for_different_seed():
    """
    计算各个seed下,基于规则的聚合方法的alignmen平均值
    """
    model = 'llama2-chat-7b'
    dataset = 'nq'
    
    model_tail = {
        'llama2-chat-7b': 'llama7b',
        'llama3-8b-instruct': 'llama8b',
        'qwen2': 'qwen2'
    }
    for mode in ['first', 'last', 'avg']:
        total_align = []
        for seed in [0, 42, 100]:
            freeform_conf = f'../share/res/{dataset}/{model}/mid_layer/zero-shot-chat/mid_layer/sample_res/pred_sample_{mode}_seed{seed}.jsonl'
            label_path = f'../share/res/{dataset}/{model}/mid_layer/zero-shot-chat/mid_layer/test_labels.pt'
            ans_path = []
            for cnt in [4, 6, 8]:
                ans_path.append(f'../share/res/{dataset}-mc/{model}/mid_layer/zero-shot-wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/mid_layer/sample_res/pred_sample_{mode}_seed{seed}.jsonl')
            all_list, gt_conf = alignment_match_for_different_files(freeform_conf, ans_path, label_path)
            seed_align = rule_based_cooperate(all_list, gt_conf)
            total_align.append(seed_align)
        print(f'mode {mode}: {sum(total_align)/len(total_align)}')

if __name__ == '__main__':            
    rule_based_alignment_for_different_seed()

    


