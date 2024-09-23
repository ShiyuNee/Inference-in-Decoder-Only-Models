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

def compute_consistency_of_different_layers(data, acc_list, net):
    net.to("cuda")
    layers = [23, 24, 25, 26, 27, 28, 29, 30, 32]
    all_layers = len(layers)
    count_num = [[] for _ in range(all_layers)]
    acc = [[] for _ in range(all_layers)]
    acc_chunk = [[[] for _ in range(all_layers)] for _ in range(10)]
    prob_list = [[] for _ in range(all_layers)]
    prob_list_right = [[] for _ in range(all_layers)]
    prob_list_wrong = [[] for _ in range(all_layers)]
    for idx in tqdm(range(len(data))):
        sample = data[idx]
        sample = net(sample.to("cuda")) # 32000
        sample = nn.Softmax(dim=1)(sample)

        probs = sample[layers]
        max_probs, max_idx = torch.max(probs, dim=1)
        final_choice = max_idx[-1]
        early_choice = max_idx[:-1]
        count = torch.sum(early_choice == final_choice).item()
        # 统计
        count_num[count].append(1)
        acc[count].append(acc_list[idx].item())
        acc_chunk[int(max_probs[-1].item() * 10)][count].append(acc_list[idx].item())
        prob_list[count].append(max_probs[-1].item())
        if acc_list[idx] == 1:
            prob_list_right[count].append(max_probs[-1].item())
        else:
            prob_list_wrong[count].append(max_probs[-1].item())
    new_acc = []
    new_probs = []
    new_probs_right = []
    new_probs_wrong = []
    for idx in range(len(acc)):
        temp_acc = 0 if len(acc[idx]) == 0 else round(sum(acc[idx]) / len(acc[idx]), 4)
        temp_prob = 0 if len(prob_list[idx]) == 0 else round(sum(prob_list[idx]) / len(prob_list[idx]), 4)
        temp_prob_right = 0 if len(prob_list_right[idx]) == 0 else round(sum(prob_list_right[idx]) / len(prob_list_right[idx]), 4)
        temp_prob_wrong = 0 if len(prob_list_wrong[idx]) == 0 else round(sum(prob_list_wrong[idx]) / len(prob_list_wrong[idx]), 4)
        new_acc.append(temp_acc)
        new_probs.append(temp_prob)
        new_probs_right.append(temp_prob_right)
        new_probs_wrong.append(temp_prob_wrong)
    print(f'count: {[len(item) for item in count_num]}')
    print(f'acc: {new_acc}')
    print(f'probs: {new_probs}')
    print(f'probs_right: {new_probs_right}')
    print(f'probs_wrong: {new_probs_wrong}')
    print(f'acc chunk-----------------------------------------------------------------------------------------------')
    save_acc_chunk = []
    save_chunk_count = []
    for item in acc_chunk:
        print(f'count: {[len(t) for t in item]}')
        print(f'avg acc: {[round(sum(t) / (len(t) + 1e-9), 4) for t in item]}')
        save_acc_chunk.append([round(sum(t) / (len(t) + 1e-9), 4) for t in item])
        save_chunk_count.append([len(t) for t in item])
    df = pd.DataFrame(save_acc_chunk)
    df_count = pd.DataFrame(save_chunk_count)
    file_path = f'./data/mmlu/acc_chunk.xlsx'
    count_file_path = f'./data/mmlu/acc_chunk_count.xlsx'
    df.to_excel(file_path, index=False, header=False)
    df_count.to_excel(count_file_path, index=False, header=False)

def run_choose_layer():
    net = nn.Linear(4096, 32000, bias=False)
    dic = {'weight': torch.load('./models/llama2-7B-chat_lm_head.pt')}
    net.load_state_dict(dic)
    data = torch.load('./data/mmlu/all_layers/data.pt')
    acc_list = torch.load('./data/mmlu/all_layers/labels.pt')
    compute_consistency_of_different_layers(data, acc_list, net)

def collect_attn_weights(dir):
    # 保存所有数据的attn_weights为一个tensor
    cnt = 0
    idx_list = []
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    print(f'subjects count: {len(paths)}')
    for item in tqdm(paths):
        data = read_json(os.path.join(dir, item))
        for item in data:
            temp_attn = torch.tensor(item['attn_weights'])
            print(temp_attn[0])
            temp_attn = temp_attn[:, temp_attn[0] != 0.0] # 对pad的attn weight为0,不考虑pad部分
            temp_attn = temp_attn.transpose(0,1) # 转换成可以合并为tensor的维度
            final_tensor = temp_attn if cnt == 0 else torch.cat((final_tensor, temp_attn), dim=0)
            cnt += 1
            idx_list.append(final_tensor.shape[0])
    print(final_tensor.shape)
    print(len(idx_list))
    idx_list = torch.tensor(idx_list)
    torch.save(idx_list, './data/all_layers/attn_weights_idx.pt')
    torch.save(final_tensor, './data/all_layers/attn_weights.pt')

def arrange_attn_weights(data_path, idx_path, label_path):
    """
    # *统计attn_weights
    # attn_path = './data/mmlu/all_layers/attn_weights.pt'
    # attn_idx_path = './data/mmlu/all_layers/attn_weights_idx.pt'
    # label_path = './data/mmlu/all_layers/labels.pt'
    # arrange_attn_weights(attn_path, attn_idx_path, label_path)
    """
    data = torch.load(data_path)
    idx_list = torch.load(idx_path)
    labels = torch.load(label_path)
    input_len = {
        'right': [],
        'wrong': []
    }
    entropy = {
        'right': [],
        'wrong': []
    }
    all_entropy = []
    all_acc = []
    all_len = []
    begin_idx = 0
    end_idx = len(idx_list)
    for idx in tqdm(range(len(idx_list))):
        end_idx = idx_list[idx]
        temp_data = data[begin_idx:end_idx].transpose(0, 1).to('cuda') # (32, seq_len)
        row_sum = temp_data.sum(dim=1, keepdim=True)
        temp_data = temp_data
        seq_len = temp_data.shape[-1]
        all_len.append(seq_len)
        begin_idx = end_idx
        temp_entropy = -torch.sum(temp_data * torch.log(temp_data), dim=-1, keepdim=True).transpose(0,1) # (1, 32)
        # temp_entropy = temp_entropy / seq_len
        all_entropy.append(temp_entropy[0][-1].item())
        all_acc.append(labels[idx])
        temp_type = 'wrong' if labels[idx] == 0 else 'right'
        input_len[temp_type].append(seq_len)
        entropy[temp_type] = temp_entropy if entropy[temp_type] == [] else torch.cat((entropy[temp_type], temp_entropy), dim=0)
    print(entropy['right'].shape)
    print(f"acc=1, count: {len(input_len['right'])}, entropy: {torch.mean(entropy['right'], dim=0)}, ave_len: {sum(input_len['right'])/len(input_len['right'])}")
    print(f"acc=0, count: {len(input_len['wrong'])}, entropy: {torch.mean(entropy['wrong'], dim=0)}, ave_len: {sum(input_len['wrong'])/len(input_len['wrong'])}")
    sorted_indices = sorted(range(len(all_entropy)), key=lambda i: all_entropy[i])
    # sorted_indices = sorted(range(len(all_len)), key=lambda i: all_len[i])
    sorted_entrop = [all_entropy[i] for i in sorted_indices]
    sorted_acc = [all_acc[i].item() for i in sorted_indices]
    sorted_len = [all_len[i] for i in sorted_indices]
    begin_thre = 0
    for thre in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        thre_idx = int(thre * len(all_acc))
        data_count = thre_idx - begin_thre
        print(f'thre: {thre}, count: {data_count}, avg len: {round(sum(sorted_len[begin_thre:thre_idx]) / (data_count), 3)}, avg entropy: {round(sum(sorted_entrop[begin_thre:thre_idx]) / (data_count), 3)}, avg acc: {round(sum(sorted_acc[begin_thre:thre_idx]) / (data_count), 3)}')
        begin_thre = thre_idx
    for temp_type in ['right', 'wrong']:
        save_entropy_right = torch.mean(entropy[temp_type], dim=0).tolist()
        save_pd = [[round(item, 4) for item in save_entropy_right]]
        df = pd.DataFrame(save_pd)
        file_path = f'./data/{temp_type}.xlsx'
        df.to_excel(file_path, index=False, header=False)
        print(f'successfully save to {file_path}')

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

def clean_ans(data):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    stop_words = list(stopwords.words('english'))
    stop_words.append('sure')
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    lemmatizer = WordNetLemmatizer()
    # print(stop_words)
    data = read_json('./data/nq/llama2-chat-7b/mid_layer/zero-shot-chat/nq_test_llama7b_tokens_mid_layer.jsonl')

    for idx in range(len(data))[:30]:
        # ans = word_tokenize(data[idx]['Res'])
        # print(data[idx]['Res'])
        ans = ("".join([char for char in data[idx]['Res'] if char not in string.punctuation])).split(' ')
        question_words = [lemmatizer.lemmatize(item, wordnet.VERB) for item in word_tokenize(data[idx]['question'].lower())]
        if len(ans) <= 3:
            filtered_sentence = ans
        else:
            filtered_sentence = [word for word in ans if word.lower() not in stop_words and lemmatizer.lemmatize(word.lower(), wordnet.VERB) not in question_words]
        print(f"question: {data[idx]['question']}, answer: {filtered_sentence}")

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

             

if __name__ == '__main__':
    # model = 'llama3-8b-instruct'
    # chat_mode = 'zero-shot-none'
    # dataset = 'hq'
    # get_res_for_different_seed(f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/mid_layer/res/')
    # compute_acc(f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/{dataset}-test-gene-none.jsonl')
    for dataset in ['nq', 'hq']:
        for chat_mode in ['zero-shot-none']:
            train_sample_path = f'../share/res/{dataset}-mc/qwen2/mid_layer/{chat_mode}/{dataset}-train-none-choice.jsonl'
            sample_training_data_for_random_mc(train_sample_path, 1)



    


        






