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

class HiddenData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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
    prepare_mode_data_for_nq(dir, 'mid')

    

