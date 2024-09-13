import torch
from torch import nn
import json
data = torch.load('./mlp_dev_pred.pt')
with open('test.json') as file:
    test_idx = json.load(file)
targets = torch.load('./data/mmlu/all_layers/labels.pt')[test_idx]
acc = torch.sum(data == targets, dim=1) / data.shape[-1]
print(f'acc: {acc}')

# 不同的部分
print(torch.sum(data[-1] == data, dim=1) / data.shape[-1])
acc_chunk = [[] for _ in range(18)]

# for idx in range(data.shape[-1]):
#     count = torch.sum(data[22:, idx] == 1)
#     acc_chunk[count].append(targets[idx] == 1)
# print([sum(item) / (len(item) + 1e-9) for item in acc_chunk])
    

diff_idx = data[-1] != data
# for idx in range(len(diff_idx)):
#     # diff_idx[idx] = [int(item) for item in diff_idx[idx]]
#     diff_data = data[idx, diff_idx[idx]]
#     diff_target = targets[diff_idx[idx]]
#     print(f'count: {len(diff_target)}')
#     print(sum(diff_data == diff_target) / len(diff_target))





