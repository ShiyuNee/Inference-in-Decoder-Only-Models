import json

with open('ood.json') as file:
    ood_idx = json.load(file)
with open('train.json') as file:
    train_idx = json.load(file)
with open('test.json') as file:
    test_idx = json.load(file)

cnt = 0
for item in ood_idx:
    if item in train_idx or item in test_idx:
        cnt += 1
print(cnt)