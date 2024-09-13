import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from collect import read_json, write_jsonl
import json
import os
import random
# tokenizer = AutoTokenizer.from_pretrained("../llama2-7B-chat")
# lm_head = nn.Linear(4096, 32000, bias=False)
# dic = {'weight': torch.load('./models/llama2-7B-chat_lm_head.pt')}
# lm_head.load_state_dict(dic)

class HiddenData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def split_data(data_path, label_path, need_layers):
    # 将mmlu后面的任务作为ood test set，前面的任务切分为train和test set
    origin_data = torch.load(data_path)
    all_data = torch.load(data_path)[:, need_layers, :] if len(origin_data.shape) == 3 else torch.load(data_path)
    labels = torch.load(label_path) # true/false
    new_labels = torch.zeros((len(labels), 2))
    for idx in range(len(labels)):
        new_labels[idx][int(labels[idx])] = 1
    with open('ood.json') as file:
        ood_idx = json.load(file)
    with open('train.json') as file:
        train_idx = json.load(file)
    with open('test.json') as file:
        test_idx = json.load(file)
    train_data = all_data[train_idx]
    train_labels = new_labels[train_idx]
    test_data = all_data[test_idx]
    test_labels = new_labels[test_idx]
    ood_data = all_data[ood_idx]
    ood_labels = new_labels[ood_idx]
    print(f'train data: {train_data.shape}')
    print(f'train labels: {train_labels.shape}')
    print(f'test data: {test_data.shape}')
    print(f'ood data: {ood_data.shape}')
    return train_data, train_labels, test_data, test_labels, ood_data, ood_labels

def split_data_for_generation(data_path, label_path, need_layers):
    data = {
        'train': [],
        'dev': [],
        'test': []
    }
    labels = {
        'train': [],
        'dev': [],
        'test': []
    }
    for mode in ['train', 'dev', 'test']:  
        if mode != 'train':
            temp_data_path = data_path.replace('train', mode).replace('sample_', '')
            temp_label_path = label_path.replace('train', mode).replace('sample_', '')
        else:
            temp_data_path = data_path
            temp_label_path = label_path
        temp_data = torch.load(temp_data_path)
        data[mode] = torch.load(temp_data_path)[:, need_layers, :] if len(temp_data.shape) == 3 else torch.load(temp_data_path)
        temp_labels = torch.load(temp_label_path) # true/false
        labels[mode] = torch.zeros((len(temp_labels), 2))
        for idx in range(len(temp_labels)):
            labels[mode][idx][int(temp_labels[idx])] = 1
    train_data = data['train']
    train_labels = labels['train']
    dev_data = data['dev']
    dev_labels = labels['dev']
    test_data = data['test']
    test_labels = labels['test']
    print(f'train data: {train_data.shape}')
    print(f'train labels: {train_labels.shape}')
    print(f'dev data: {dev_data.shape}')
    print(f'test data: {test_data.shape}')
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels

def split_data_for_mmlu(data_path, label_path, need_layers):
    origin_data = torch.load(data_path)
    all_data = torch.load(data_path)[:, need_layers, :] if len(origin_data.shape) == 3 else torch.load(data_path)
    labels = torch.load(label_path) # true/false
    new_labels = torch.zeros((len(labels), 2))
    for idx in range(len(labels)):
        new_labels[idx][int(labels[idx])] = 1
    # train_sample_idx = random.sample(range(len(all_data)), int(len(all_data) * 0.5))
    # remain_idx = [item for item in range(len(all_data)) if item not in train_sample_idx]
    # dev_sample_idx = random.sample(remain_idx, int(len(remain_idx) * 0.5))
    # test_sample_idx = [item for item in range(len(all_data)) if item not in train_sample_idx and item not in dev_sample_idx]
    train_sample_idx = read_json('mmlu_train.jsonl')
    dev_sample_idx = read_json('mmlu_dev.jsonl')
    test_sample_idx = read_json('mmlu_test.jsonl')

    train_data = all_data[train_sample_idx]
    train_label = new_labels[train_sample_idx]
    dev_data = all_data[dev_sample_idx]
    dev_label = new_labels[dev_sample_idx]
    test_data = all_data[test_sample_idx]
    test_label = new_labels[test_sample_idx]

    print(f'train data: {train_data.shape}')
    print(f'train labels: {train_label.shape}')
    print(f'dev data: {dev_data.shape}')
    print(f'test data: {test_data.shape}')
    return train_data, train_label, dev_data, dev_label, test_data, test_label

def get_contrastive_layer(data):
    # print(mature_logits.shape)
    # print(candidate_logits.shape)
    # Pick the less like layer to contrast with
    # 1. Stacking all premature_layers into a new dimension
    # 2. Calculate the softmax values for mature_layer and all premature_layers
    new_data = []
    logits = lm_head(data)
    softmax_mature_layer = nn.Softmax(dim=-1)(logits[:, -1, :])  # shape: (batch_size, num_features)
    # print(softmax_mature_layer.shape)
    softmax_premature_layers = nn.Softmax(dim=-1)(logits[:, :32, :])  # shape: (batch_size, num_premature_layers, num_features)
    # print(softmax_premature_layers.shape)

    # 3. Calculate M, the average distribution
    M = 0.5 * (softmax_mature_layer[:, None, :] + softmax_premature_layers)  # shape: (batch_size, num_premature_layers, num_features)

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(logits[:, -1, :], dim=-1)  # shape: (batch_size, num_features)
    log_softmax_premature_layers = F.log_softmax(logits[:, :32, :], dim=-1) # shape: (batch_size, num_premature_layers, num_features)

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer[:, None, :], M, reduction='none').mean(-1)  # shape: (batch_size, num_premature_layers)
    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (batch_size, num_premature_layers)
    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)
    js_divs = js_divs
    # print(js_divs.shape)
    # 6. Reduce the batchmean
    # js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
    js_divs = torch.argmax(js_divs, dim=1).tolist()
    for idx in range(len(js_divs)):
        new_data.append([data[idx][-1].tolist(), data[idx][int(js_divs[idx])].tolist()])
    return torch.tensor(new_data)

def prepare_data(path):
    data = read_json(path)
    modes = ['first', 'last', 'min', 'avg', 'ans', 'dim_max', 'dim_min']
    mode = ""
    for item in modes:
        if item in path:
            mode = item
    
    out_data_path = '/'.join(path.split('/')[:-1]) + '/' + mode + '_data.pt'
    out_label_path = '/'.join(path.split('/')[:-1]) + '/' + mode + '_label.pt'
    print(out_data_path)
    hidden_data = []
    labels = []
    for item in data:
        hidden_data.append(item['hidden_states'])
        labels.append(item['has_answer'])
    hidden_data = torch.tensor(hidden_data)
    labels = torch.tensor(labels)
    torch.save(hidden_data, out_data_path)
    torch.save(labels, out_label_path)

def get_train_dev_test_data():
    base_dir = './data/nq/mid_layers/llama3-8b-instruct/test/'
    paths = os.listdir(base_dir)
    for item in paths:
        file_path = base_dir + item
        prepare_data(file_path)

def prepare_mode_data(path, hidden_modes, mode_hidden_states={}, labels=[]):
    """
    得到一个文件内所有数据的所有mode的hidden state,以及labels
    """
    data = read_json(path)
    if mode_hidden_states == {}:
        for mode in hidden_modes:
            mode_hidden_states[mode] = []

    for item in data:
        labels.append(item['has_answer'])
        for mode in hidden_modes:
            hidden_state = item['hidden_states'][mode]
            if len(hidden_state) != 0:
                mode_hidden_states[mode].append(hidden_state)
    return mode_hidden_states, labels

def prepare_mode_data_for_dir(dir, mode='mid'):
    """
    得到mmlu的所有mode得到的hidden states
    """
    paths = [item for item in os.listdir(dir) if '.jsonl' in item]
    hidden_states = {}
    labels = []
    hidden_modes = ['first', 'last', 'avg']
    for path in paths:
        file_path = dir + path
        hidden_states, labels = prepare_mode_data(file_path, hidden_modes, hidden_states, labels)
    print(f'count: {len(labels)}')

    if not os.path.exists(dir + mode + '_layer/'):
        os.mkdir(dir + mode + '_layer/')
    for k, v in hidden_states.items():
        print(f'{k}: {len(v)}')
        if len(v) != 0:
            out_path = dir + mode + '_layer/' + k + '.pt'
            torch.save(torch.tensor(v), out_path)
    out_label = dir + mode + '_layer/labels.pt'
    torch.save(torch.tensor(labels), out_label)    

def prepare_mode_data_for_nq(dir, mode):
    paths = [item for item in os.listdir(dir) if '.jsonl' in item]
    hidden_modes = ['first', 'last', 'avg']
    for path in paths:
        file_path = dir + path
        hidden_states = {}
        labels = []
        hidden_states, labels = prepare_mode_data(file_path, hidden_modes, hidden_states, labels)
        print(f'count: {len(labels)}')
        if not os.path.exists(dir + mode + '_layer/'):
            os.mkdir(dir + mode + '_layer/')
        for k, v in hidden_states.items():
            print(f'{k}: {len(v)}')
            if len(v) != 0:
                out_path = dir + mode + '_layer/' + k + '_' +  path.replace('-', '_').split('_')[1] + '.pt'
                torch.save(torch.tensor(v), out_path)
        out_label = dir + mode + '_layer/'+ path.replace('-', '_').split('_')[1] + '_labels.pt'
        torch.save(torch.tensor(labels), out_label) 

def prepare_sample_train_data(train_path):
    dir = '/'.join(train_path.split('/')[:-1]) + '/'
    mode = 'mid'
    hidden_states, labels = prepare_mode_data(train_path, ['first', 'avg', 'last'], {}, [])
    for k, v in hidden_states.items():
        print(f'{k}: {len(v)}')
        if len(v) != 0:
            out_path = dir + mode + '_layer/sample_' + k + '_train.pt'
            torch.save(torch.tensor(v), out_path)
    out_label = dir + mode + '_layer/sample_train_labels.pt'
    torch.save(torch.tensor(labels), out_label)   

if __name__ == "__main__":
    dir = '../share/res/nq/qwen2/mid_layer/zero-shot-chat/' 

    # prepare_mode_data_for_dir(dir, 'mid')
    prepare_mode_data_for_nq(dir, 'mid')

    

