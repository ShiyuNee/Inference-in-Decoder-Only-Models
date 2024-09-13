from sklearn.metrics import roc_auc_score
import os
import numpy as np
from collect import read_json
from tqdm import tqdm


def compute_auroc(probs, labels):
    ood_idx = read_json('./ood.json')
    probs = np.array(probs)
    labels = np.array(labels)
    auroc = roc_auc_score(labels[ood_idx][0], probs[ood_idx][0])
    print(f'The AUROC is: {auroc}')

def arrange_probs(dir):
    choices = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    labels = []
    probs = []
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            labels.append(sample['has_answer'])
            probs.append(sample['Log_p']['token probs'][choices[sample['Res']]])
    return probs, labels


probs, labels = arrange_probs('./data/mmlu/zero-shot')
compute_auroc(probs, labels)



