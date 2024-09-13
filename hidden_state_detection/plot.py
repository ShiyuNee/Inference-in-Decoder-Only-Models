import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import random
from torch import nn
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from matplotlib.lines import Line2D
from collect import split_all_data_to_task_data
from convert import convert_tokens_to_consistency, prepare_data_for_plot
random.seed(0)
# 创建示例数据

net = nn.Linear(4096, 32000, bias=False)
dic = {'weight': torch.load('./models/llama2-7B-chat_lm_head.pt')}
net.load_state_dict(dic)
tokenizer = AutoTokenizer.from_pretrained("../llama2-7B-chat")

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def get_data(path, net, tokenizer):
    top_k = 10
    labels = []
    data = []
    tokens = []
    all_tokens = []
    sub_data = read_json(path)
    all_idx = list(range(len(sub_data)))
    for idx in range(len(sub_data)):
        sample = sub_data[idx]
        temp_tokens = []
        hidden_states = torch.tensor(sample['hidden_states'])
        scores = net(hidden_states)
        scores = nn.Softmax(dim=1)(scores)
        top_values, top_idx = torch.topk(scores[-1], k=top_k)
        top_logits = scores[:, top_idx].tolist()
        _, all_top_idx = torch.topk(scores, k=top_k)
        for temp_idx in all_top_idx:
            temp_tokens.append(tokenizer.convert_ids_to_tokens(temp_idx))
        all_tokens.append(temp_tokens)
        tokens.append(tokenizer.convert_ids_to_tokens(top_idx))
        data.append(top_logits)
        labels.append(sample['has_answer'])
    return torch.tensor(data), torch.tensor(labels), tokens, all_tokens

def plot_line(data, labels, mode, task, dataset):
    """
    data: 输出token在每层的概率
    labels: 输出token与ground truth是否相同
    mode: sample与mean, sample为采样一些样本绘制, mean为绘制所有样本的平均值
    task: mmlu与tq, 影响保存图片的路径
    """
    x = list(range(33)[1:])
    colors = ['red', 'blue']
    color_labels = ['right', 'wrong']

    def plot_sample(data, labels):
        right_count = 10
        wrong_count = 10
        for idx in range(len(data)):
            probs = data[idx].tolist()
            color = 'red' if labels[idx] == 1 else 'blue'
            if labels[idx] == 1 and right_count > 0:
                right_count -= 1
                plt.plot(x, probs, color=color, lw=0.5)
            if labels[idx] == 0 and wrong_count > 0:
                wrong_count -= 1
                plt.plot(x, probs, color=color, lw=0.5)
            
        custom_lines = [Line2D([0], [0], color=colors[i], lw=1, label=color_labels[i]) for i in range(len(colors))]
        plt.legend(handles=custom_lines)
        plt.title(dataset)
        plt.savefig(f'./plot/{task}/{dataset}_line.png')
        plt.close()
    
    def plot_mean(data, labels):
        """
        绘制每一层prob/consistency的平均值
        """
        right_mean = [0 for _ in range(32)]
        wrong_mean = [0 for _ in range(32)]
        right_cnt = 0
        wrong_cnt = 0
        for idx in range(len(data)):
            probs = data[idx].view(-1).tolist()
            if labels[idx] == 1:
                right_mean = [right_mean[i] + probs[i] for i in range(32)]
                right_cnt += 1
            else:
                wrong_mean = [wrong_mean[i] + probs[i] for i in range(32)]
                wrong_cnt += 1
        right_mean = [item / right_cnt for item in right_mean]
        wrong_mean = [item / wrong_cnt for item in wrong_mean]
        plt.plot(x, right_mean, color='red', label='right')
        plt.plot(x, wrong_mean, color='blue', label='wrong')
        plt.title(dataset)
        plt.legend()
        plt.savefig(f'./plot/{task}/{dataset}_line_mean.png')
        plt.close()
    if mode == 'mean':
        plot_mean(data, labels)
    elif mode == 'sample':
        plot_sample(data, labels)
    else:
        raise ValueError(f'Give the wrong mode: {mode}')

def get_attn_weights(path, tokenizer):
    data = []
    tokens = []
    labels = []
    sub_data = read_json(path)
    top_k = 32
    for idx in range(len(sub_data)):
        sample = sub_data[idx]
        attn_weights = torch.tensor(sample['attn_weights'])
        attn_weights = attn_weights[:, attn_weights[0] != 0.0]
        attn_weights = attn_weights[:, 1:-2]
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True)
        # top_values, top_idx = torch.topk(nn.Softmax(dim=0)(attn_weights[-1]), k=top_k)
        labels.append(sample['has_answer'])
        data.append(attn_weights)
        temp_tokens = tokenizer.convert_ids_to_tokens(tokenizer(sample['qa_prompt'])['input_ids'])[1:-2]
        tokens.append(temp_tokens)
    return data, torch.tensor(labels), tokens

def plot_map(data, label, subject, tokens, all_tokens, plot_dir):
    """
    分别绘制做对/做错的问题答案token相关信息: 最后一层top-k token以及在每层的概率,每层top-k token
    Input:
        - data: 最后一层top-k token概率信息 (N, layers, k)
        - labels: 答案是否正确
        - tokens: 最后一层top-k tokens
        - all_tokens: 每层top-k tokens
    Return:
        - 路径: ./{plot_dir}/subject/xxx.png
    """
    base_dir = subject
    if not os.path.exists(f'./{plot_dir}/{base_dir}'):
        os.makedirs(f'./{plot_dir}/{base_dir}')
    right_data = []
    wrong_data = []
    wrong_tokens = []
    right_tokens = []
    right_all_tokens, wrong_all_tokens = [], []
    plot_start_layer = 10

    for idx in range(len(label)):
        if int(label[idx]) == 1:
            if type(data[idx]) == list:
                right_data.append(data[idx])
            else:
                right_data.append(data[idx].numpy())
            right_tokens.append(tokens[idx])
            right_all_tokens.append(all_tokens[idx])
        else:
            if type(data[idx]) == list:
                wrong_data.append(data[idx])
            else:
                wrong_data.append(data[idx].numpy())
            wrong_tokens.append(tokens[idx])
            wrong_all_tokens.append(all_tokens[idx])
    right_idx = random.sample(range(len(right_data)), 10)
    wrong_idx = random.sample(range(len(wrong_data)), 10)

    for mode in ["right", "wrong"]:
        plt_data= [] # 最后一层top-k token概率
        plt_tokens = [] # 最后一层top-k token
        plt_all_tokens = [] # 每一层top-k token
        if mode == "right":
            for idx in right_idx:
                plt_data.append(right_data[idx])
                plt_tokens.append(right_tokens[idx])
                plt_all_tokens.append(right_all_tokens[idx])
        else:
            for idx in wrong_idx:
                plt_data.append(wrong_data[idx])
                plt_tokens.append(wrong_tokens[idx])
                plt_all_tokens.append(wrong_all_tokens[idx])
        # 绘制最后一层top-k token在每一层的概率
        for idx in range(len(plt_data)):
            plt.figure(figsize=(12, 12))  # 调整图表大小
            ax = sns.heatmap(plt_data[idx][plot_start_layer:], annot=True, fmt=".2f", cmap='YlGnBu', vmin=0, vmax=1.0)
            try:
                plt_ids2tokens = tokenizer.convert_ids_to_tokens(plt_tokens[idx])
            except:
                plt_ids2tokens = plt_tokens[idx]
            print(f'plt tokens: {plt_ids2tokens}')
            ax.set_xticklabels(plt_ids2tokens, rotation=90)
            y_labels = range(len(plt_data[idx]))[plot_start_layer:]
            ax.set_yticklabels([str(item) for item in y_labels])
            plt.savefig(f'./{plot_dir}/{base_dir}/{mode}_{idx}.png')
            plt.close()
        
        for idx in range(len(plt_all_tokens)): # 绘制每一层top-k tokens
            try:
                plt_all_ids2tokens = [tokenizer.convert_ids_to_tokens(item) for item in plt_all_tokens[idx]]
            except:
                plt_all_ids2tokens = plt_all_tokens[idx]
            # print(plt_all_ids2tokens)
            data = np.array(plt_all_ids2tokens)
            
            # 设置图形大小
            fig, ax = plt.subplots(figsize=(12, 12))
            # 创建一个空的热图作为背景
            ax.imshow(np.zeros(data.shape), cmap='Wistia', aspect='auto')

            # 在图中添加文本
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, data[i, j], ha='center', va='center', color='black')

            # 设置坐标轴
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_xticklabels(np.arange(1, data.shape[1] + 1))
            ax.set_yticklabels(np.arange(1, data.shape[0] + 1))

            # 设置标题和标签
            plt.title('String Matrix')
            plt.xlabel('Column')
            plt.ylabel('Row')

            # 隐藏网格线
            ax.grid(False)

            # 显示图形
            plt.savefig(f'./{plot_dir}/{base_dir}/{mode}_{idx}_all_tokens.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

def plot_line_for_mmlu(plot_type):
    """
    绘制每个subtask的概率/consistency曲线
    """
    label_path = './data/mmlu/all_layers/labels.pt'
    dir = './data/mmlu/zero-shot'
    mode = 'mean'
    if plot_type == 'prob':
        data_path = './data/mmlu/all_layers/data_logits_1.pt' # 所有数据
        task_data, task_label, task_names = split_all_data_to_task_data(torch.load(data_path), torch.load(label_path), dir) # 将数据按task划分

    elif plot_type == 'consistency':
        data_path = './data/mmlu/all_layers/token_ids_10_diff_layers.pt'
        task_data, task_label, task_names = split_all_data_to_task_data(convert_tokens_to_consistency(torch.load(data_path)), torch.load(label_path), dir)
    else:
        raise ValueError(f'Specify the wrong plot_type: {plot_type}')
    
    task_names = [item.replace('.jsonl', '') for item in task_names] if plot_type == 'prob' else [f"{item.replace('.jsonl', '')}_{plot_type}" for item in task_names]
    for idx in range(len(task_data)):
        plot_line(task_data[idx], task_label[idx], mode, 'mmlu', task_names[idx])
    
def plot_line_for_tq(plot_type):
    label_path = './data/tq/all_layers/labels.pt' # acc for each sample
    labels = torch.load(label_path)
    mode='mean'
    if plot_type == 'prob':
        data_path = './data/tq/all_layers/data_logits_1.pt'
        data = torch.load(data_path)
    elif plot_type == 'consistency':
        data_path = './data/tq/all_layers/token_ids_10_diff_layers.pt'
        data = torch.load(data_path)
        data = convert_tokens_to_consistency(data)
    else:
        raise ValueError(f'Specify the wrong plot_type: {plot_type}')

    dataset = 'TruthfulQA' if plot_type =='prob' else f'TruthfulQA_{plot_type}'
    plot_line(data, labels, mode, 'tq', dataset)


if __name__ == '__main__':
    data, labels, tokens, all_tokens = prepare_data_for_plot('./data/nq/nq_test_llama7b_every_tokens.jsonl')
    plot_map(data, labels, 'nq/top-1', tokens, all_tokens, './plot/nq/map')
    


