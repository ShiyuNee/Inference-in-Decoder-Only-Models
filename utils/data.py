import json
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils.prompt import get_prompt
import pandas as pd
import os

class QADataset(Dataset):
    """
    Open-domain generation dataset
    """
    def __init__(self, args):
        self.data = self.read(args.source)
        self.prompts = []
        self.idxs = []
        self.args = args
        self.get_prompted_data()

    def read(self, path):
        qa_data = []
        f = open(path, 'r', encoding='utf-8')
        for line in f.readlines():
            qa_data.append(json.loads(line))
        return qa_data
    
    def get_prompted_data(self):
        for idx in range(len(self.data)):
            if 'info' not in self.data[idx]:
                self.idxs.append(idx)
                self.prompts.append(get_prompt(self.data[idx], self.args)) 
        for item in self.prompts[:5]:
            print(f'example: {item}')

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]
    
class MCDataset(Dataset):
    """
    Multi-choice dataset
    """
    # generate input for the given subject
    def __init__(self, args, subject):
        self.args = args
        self.subject = subject
        self.data = self.read('test')
        self.idxs = range(len(self.data))
        self.dev_data = self.read('dev') if self.args.n_shot != 0 else []
        self.get_choice_count()
        self.prompts = []
        self.get_prompted_data()
    
    def get_choice_count(self):
        all_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.choice_cnt = len(self.data[0]) - 2
        self.choices = all_choices[:self.choice_cnt]


    def read(self, mode='test'):
        mmlu_data = pd.read_csv(os.path.join(self.args.source, mode, self.subject + f"_{mode}.csv"), header=None).to_numpy() # no header
        return mmlu_data
    
    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    
    def format_example(self, data, idx, include_answer=True):
        prompt = data[idx][0] # question
        k = len(data[idx]) - 2 # count of choices
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], data[idx][j+1]) # append each candidate answer
        return prompt
    
    def get_prompted_data(self):
        if self.args.task == 'mmlu':
            self.args.subject = ' about' + self.format_subject(self.subject) 
        else:
            self.args.subject = ''
        for idx in range(len(self.data)):
            question = self.format_example(self.data, idx, include_answer=False)
            prompt = get_prompt({'question': question}, self.args)
            self.prompts.append(prompt)
        for item in self.prompts[:5]:
            print(f'example: {item}')
        prompt_len = []
        for item in self.prompts:
            prompt_len.append(len(item.split(' ')))
        self.avg_len = sum(prompt_len)/len(prompt_len)
        self.max_len = max(prompt_len)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]



