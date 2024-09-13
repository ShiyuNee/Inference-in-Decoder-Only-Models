from torch import nn
import os
import torch
from tqdm import tqdm
from collect import read_json, write_jsonl
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt

MODEL_PATH='./models/llama2-7B-chat_lm_head.pt'
net = nn.Linear(4096, 32000, bias=False)
top_k = 10
all_top_k=10
dic = {'weight': torch.load(MODEL_PATH)}
net.load_state_dict(dic)
tokenizer = AutoTokenizer.from_pretrained("../llama2-7B-chat")

def get_labels_tensor(data):
    labels = []
    for item in data:
        labels.append(item['has_answer'])
    return torch.tensor(labels)

def convert_hidden_to_logits(dir, task):
    """
    利用lm_head将hidden states转换成token概率
    Input: 
        - dir: 包含所有任务的列表
        - task: 任务名称
    Example:
        - data_dir = './data/tq/zero-shot-hidden'
        - task = 'tq'
    """
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    data = [] # 存储最后一层的top k token在每一层的分数
    tokens = [] # 存储最后一层的top k token
    all_data = [] # 存储每一层的top k分数
    all_tokens = [] # 存储每一层的top k tokens
    labels = []
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            hidden_states = torch.tensor(sample['hidden_states'])
            if hidden_states.shape[0] != 1:
                hidden_states = torch.tensor(sample['hidden_states'][1:]) # 不要embedding
            scores = nn.Softmax(dim=1)(net(hidden_states))
            # 最后一层top-k token, 在每一层上对应的分数
            top_values, top_idx = torch.topk(scores[-1], k=top_k) # 最后一层的top-k
            top_tokens = top_idx # 对应的tokens
            layer_scores = scores[:, top_idx]
            if top_k == 1:
                layer_scores = layer_scores.reshape(-1)

            # 每一层的top tokens以及对应的分数
            all_top_values, all_top_idx = torch.topk(scores, k=all_top_k) #tensor(num_layer, top_k)
            data.append(layer_scores.tolist())
            tokens.append(top_tokens.tolist())
            all_data.append(all_top_values.tolist())
            all_tokens.append(all_top_idx.tolist())
            labels.append(int(sample['has_answer']))
               
    data = torch.tensor(data)
    tokens = torch.tensor(tokens)
    all_data = torch.tensor(all_data)
    all_tokens = torch.tensor(all_tokens)
    labels = torch.tensor(labels)
    print(f'data shape: {data.shape}')
    print(f'tokens shape: {tokens.shape}')
    print(f'all data shape: {all_data.shape}')
    torch.save(data, f'./data/{task}/all_layers/data_logits_{top_k}.pt')
    torch.save(tokens, f'./data/{task}/all_layers/token_ids_{top_k}.pt')
    torch.save(all_data, f'./data/{task}/all_layers/data_logits_{all_top_k}_diff_layers.pt')
    torch.save(all_tokens, f'./data/{task}/all_layers/token_ids_{all_top_k}_diff_layers.pt')

def convert_tokens_to_consistency(data):
    """
    将token信息转化成层间一致性(中间层是否选择最终输出的token)
    Input:
        - data: shape(num, layers, top-k)
    Return:
        - consistency: shape(num, layers)
    Example:
        - data = torch.load('./data/mmlu/all_layers/token_ids_10_diff_layers.pt')
        - convert_tokens_to_consistency(data)
    """
    consistency = []
    for item in data:
        final_choice = item[-1][0]
        early_choice = item[:, 0]
        temp_consistency = early_choice == final_choice
        consistency.append(temp_consistency)
    consistency = torch.stack(consistency)
    return consistency

def down_dimension(data, labels):
    """
    降维绘制做对做错的样本hidden state
    Example
        - data = torch.load('./data/mmlu/all_layers/data.pt')
        - labels = torch.load('./data/mmlu/all_layers/labels.pt')
        - down_dimension(data[:, -1, :], labels)
    """
    pac = PCA(n_components=3)
    data_3d = pac.fit_transform(data)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'red']
    for label in np.unique(labels):
        indices = labels == label
        ax.scatter(data_3d[indices, 0], data_3d[indices, 1], data_3d[indices, 2], c=colors[label], label=f'Label {label}', alpha=0.6, edgecolors='w', s=10)
    plt.legend()
    plt.savefig('./mmlu_pca.png')

def find_token_range(s, tokens):
    """
    得到tokens中组成字符串s的token的起始和终止index
    """
    # 拼接tokens以得到它们组成的完整字符串
    full_string = "" # 利用tokens拼接成字符串
    token_str = [] # 记录每个token在字符串中所占的起始和终止index
    str_cur_idx = 0
    # 找s在full_string中的所占的区间
    for token in tokens:
        full_string += token.replace('▁', ' ').lower()
        token_str.append([str_cur_idx, str_cur_idx + len(token) - 1])
        str_cur_idx = str_cur_idx + len(token)

    # 在 full_string 中查找 s 的起始位置
    str_start_index = full_string.find(s)
    if str_start_index != -1:
        str_end_idx = str_start_index + len(s) - 1
    else:
        s_words = s.split()
        str_start_index = full_string.find(s_words[0])
        if str_start_index == -1:
            print(s_words)
        return None  # 如果没找到，则返回None
    
    
    # assert full_string[str_start_index:str_end_idx+1] == s
    # 将字符的位置对应为token列表中token的位置
    token_start_idx = 0
    token_end_idx = len(tokens) - 1
    for idx in range(len(token_str)):
        item = token_str[idx]
        if str_start_index >= item[0] and str_start_index <= item[1]:
            token_start_idx = idx
        if str_end_idx >= item[0] and str_end_idx <= item[1]:
            token_end_idx = idx
    # print(f'find tokens: {tokens[token_start_idx: token_end_idx+1]}')
    return token_start_idx, token_end_idx

def find_need_idx(tokens, answers):
    assert len(tokens) == len(answers)
    cnt = 0
    for idx in range(len(tokens)):
        temp_tokens = tokenizer.convert_ids_to_tokens(tokens[idx]['Log_p']['tokens'])
        temp_answer = answers[idx]['Res'].lower()

        res = find_token_range(temp_answer, temp_tokens)
        if res == None:
            cnt += 1
            # print(f'tokens: {temp_tokens}')
            # print(f"token probs: {tokens[idx]['Log_p']['token_probs']}")
            # print(f'extracted answer: {temp_answer.split()}')
            print(f'not mathc count: {cnt}')
    
def prepare_data_for_plot(path):
    """
    prepare_data_for_plot('./data/nq/nq_test_llama7b_every_tokens.jsonl')
    """
    data = read_json(path)
    probs_for_each_layer = []
    tokens_for_all_layers = []
    generated_tokens = []
    labels = []
    for item in data:
        probs_for_each_layer.append(item['probs_for_generated_tokens'])
        tokens_for_all_layers.append(item['tokens_for_each_layer'])
        generated_tokens.append(item['tokens_for_each_layer'][-1])
        labels.append(int(item['has_answer']))
    return probs_for_each_layer, labels, generated_tokens, tokens_for_all_layers
        
        
if __name__ == '__main__':
    path = './data/nq/llama2-chat-7b/mid_layer/zero-shot-chat/nq_test_llama7b_tokens_mid_layer.jsonl'
    data = read_json(path)
    new_data = []
    for item in data:
        new_data.append({'question': item['question'], 'Res': item['Res'], 'has_answer': item['has_answer'], 'reference': item['reference']})
    out_file = '/'.join(path.split('/')[:-1]) + '/' + path.split('/')[-1].replace('_tokens_mid_layer', '')
    write_jsonl(new_data, out_file)





