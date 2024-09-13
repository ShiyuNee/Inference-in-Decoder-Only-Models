import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
from torch import nn
import json
from tqdm import tqdm

net = nn.Linear(4096, 32000, bias=False)
dic = {'weight': torch.load('./models/llama2-7B-chat_lm_head.pt')}
net.load_state_dict(dic)
net.to('cuda')
def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def prepare_data():
    # 最后一层: prob, entropy, attn_weights_entropy, input_len, token_id
    # 多层: 一致性, 多层平均概率
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    ood_data = []
    ood_label = []
    logits_data = torch.load('./data/mmlu/all_layers/data_logits_1.pt')
    train_idx = read_json('./train.json')[0]
    test_idx = read_json('./test.json')[0]
    layers = [23, 24, 25, 26, 27, 28, 29, 30, 32]
    hidden_data = torch.load('./data/mmlu/all_layers/data.pt')
    labels = torch.load('./data/mmlu/all_layers/labels.pt')
    attn_data = torch.load('./data/mmlu/all_layers/attn_weights.pt')
    attn_idx  = torch.load('./data/mmlu/all_layers/attn_weights_idx.pt')
    begin_idx = 0
    for idx in tqdm(range(len(hidden_data))):
        temp_data = []
        end_idx = attn_idx[idx]
        temp_attn = attn_data[begin_idx:end_idx].transpose(0, 1)[:, 1:-2].to('cuda') # (32, seq_len)
        seq_len = temp_attn.shape[-1]
        begin_idx = end_idx
        attn_entropy = -torch.sum(temp_attn * torch.log(temp_attn), dim=-1)
        # 最后一层attn_weights的entropy
        # 得到prob
        sample = hidden_data[idx]
        sample = net(sample.to("cuda")) # 32000
        sample = nn.Softmax(dim=1)(sample)

        probs = sample[layers]
        max_probs, max_idx = torch.max(probs, dim=1)
        final_choice = max_idx[-1]
        early_choice = max_idx[:-1]
        count = torch.sum(early_choice == final_choice).item()
        
        temp_entropy = -torch.sum(probs[-1] * torch.log(probs[-1]), dim=-1)
        # temp_data.append(temp_entropy.item())
        # temp_data.append(count)
        # temp_data.append(max_probs[-1].item())
        # temp_data = logits_data[idx].tolist()
        # temp_data.append(seq_len)
        # for attn_layer in range(32):
        #     temp_data.append(attn_entropy[attn_layer].item() / seq_len)
        if idx in train_idx:
            train_data.append(temp_data)
            train_label.append(labels[idx])
        elif idx in test_idx:
            test_data.append(temp_data)
            test_label.append(labels[idx])
        else:
            ood_data.append(temp_data)
            ood_label.append(labels[idx])
    return train_data, train_label, test_data, test_label, ood_data, ood_label

X_train, y_train, X_test, y_test, X_ood, y_ood = prepare_data()
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_ood = np.array(X_ood)
y_ood = np.array(y_ood)
print(X_train.shape)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)
ood_pred = model.predict(X_ood)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

ood_accuracy = accuracy_score(y_ood, ood_pred)
print(f"OOD Accuracy: {ood_accuracy}")

# 混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)

# # 分类报告
# report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(report)
