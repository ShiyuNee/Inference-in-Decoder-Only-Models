import time
import os
from .utils import deal_answer, deal_judge, deal_post, str2paras, deal_judge_new, has_answer
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
    
class Generater:
    def __init__(self, args):
        self.args = args
        self.model = LlamaForCausalLM.from_pretrained(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.outputs = []
        print('load generater finish.')

    def load_data(self, data):
        self.data = data
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=self.batch_size)

    def get_res(self):
        self.outputs = []
        device = torch.device('cuda')
        self.device = device
        self.model.to(device)
        for batch in tqdm(self.dataloader):
            batch = self.tokenizer(batch, return_tensors='pt', padding=True).to(device)
            input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
            outs = self.model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=self.args.max_new_tokens, 
                                       output_attentions=self.args.attn_weights, return_dict_in_generate=True, output_scores=True, output_logits=True, output_hidden_states=self.args.hidden_states, 
                                       pad_token_id=0, top_p=1.0, temperature=1)
            if self.args.task == 'mmlu' or self.args.task == 'tq':
                self.process_res_multi_choice(outs, input_ids) # 得到一个batch的结果
            else:
                self.process_res(outs, input_ids)
        print(f'len of outputs: {len(self.outputs)}')
        return self.calculate_res()
    
    def process_res(self, outs, inputs):
        """
        处理模型generate输出, 得到输出文本,每个token的概率,以及每个token的entropy
        Input:
            - outs: generate输出结果
            - inputs: generate的input_ids
        Return:
            - 输出列表, 每个元素是一个字典
            {
                'Res': 生成文本,
                'Log_p':{
                    'tokens':生成的每个token,
                    'token_probs': 生成的每个token的概率,
                    'token_entropy': 生成每个token时对应的vocab空间的entropy
                }
            }
        """
        # attention和scores都不包含输入
        attentions = outs['attentions'] # tuple(generated_token, layer) -> (batch_size, num_heads, generated_length, sequence_length)
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        input_len = inputs.shape[-1]
        bt_size = inputs.shape[0]
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        # print(f'text: {self.tokenizer.batch_decode(new_ids, skip_sepcial_tokens=True)}')
        # generate会生成到所有序列都结束,因此需要得到每个seq真正的结尾 eos_token
        text_len = new_ids.shape[-1]
        end_idx = []
        for idx in range(len(new_ids)):
            eos_idx = torch.where(new_ids[idx] == self.tokenizer.eos_token_id)[0] # 返回tuple, [0]是该元素出现位置的tensor
            if len(eos_idx) == 0: # 没有eos_token
                end_idx.append(text_len)
            else:
                end_idx.append(eos_idx[0]) # eos_token出现的第一个位置
        top_indices = [] # 存储概率最大的token_id
        top_scores = [] # 存储对应的probs
        ans_scores = [] # 存储seqs对应probs
        ans_entropy = []
        for idx in range(len(scores)): # 遍历每个token
            probs = nn.Softmax(dim=1)(scores[idx]) # batch_size, vocab_size
            cur_scores = [probs[t, new_ids[t, idx]] for t in range(bt_size)] # batch_size, 每个生成token的概率
            cur_entropy = torch.sum(-(probs * torch.log2(probs)), dim=1) # batch_size
            tmp_scores, tmp_indices = torch.max(probs, dim=1) # batch_size

            ans_scores.append(cur_scores) # seq_len, batch_size
            ans_entropy.append(cur_entropy.tolist())
            top_indices.append(tmp_indices.tolist())
            top_scores.append(tmp_scores.tolist())
        top_indices = torch.tensor(top_indices, dtype=torch.int64).t()
        top_scores = torch.tensor(top_scores).t()
        ans_scores = torch.tensor(ans_scores).t()
        ans_entropy = torch.tensor(ans_entropy).t()
        for bt in range(bt_size):
            print(f'ans: {self.tokenizer.decode(new_ids[bt][:end_idx[bt]])}')
            self.outputs.append({
                'Res': self.tokenizer.decode(new_ids[bt][:end_idx[bt]]).strip(),
                'Log_p':{
                    'tokens': new_ids[bt][:end_idx[bt]].tolist(),
                    'token_probs': ans_scores[bt][:end_idx[bt]].tolist(),
                    'token_entropy': ans_entropy[bt][:end_idx[bt]].tolist()
                }
            })

    def process_res_multi_choice(self, outs, inputs):
        """
        对多选问题,得到在输出的token上的结果,概率,entropy,hidden_state等信息
        Input:
            - outs:
            - inputs:
        Return:
            - 
        """
        choices = ['A', 'B', 'C', 'D']
        out_idx = 0
        # get attn_weights when generating the first token
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        print(f'text: {self.tokenizer.batch_decode(seqs[:, inputs.shape[-1]:], skip_sepcial_tokens=True)}')

        choices_idx = self.tokenizer(choices)['input_ids']
        choices_idx = [item[1] for item in choices_idx] # <s> when idx=0
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        bt_size = inputs.shape[0]
        choices_probs = nn.Softmax(dim=1)(scores[out_idx][:, choices_idx]) #仅考虑选项的概率
        probs = nn.Softmax(dim=1)(scores[out_idx])
        next_token_probs = probs[:, choices_idx] # batch_size, 4
        entropy = torch.sum(-(probs * torch.log2(probs)), dim=1) # batch_size
        choices_entropy = torch.sum(-(choices_probs * torch.log2(choices_probs)), dim=1)
        # print(f'next token probs: {next_token_probs}')
        max_scores, max_indices = torch.max(next_token_probs, dim=1)
        if self.args.attn_weights: # 生成的第一个token, 最后一层, 所有头的attn_weights
            attentions = outs['attentions'][0][-1][:, :, -1] # bs, head_num, seq_len(input_len)
        if self.args.hidden_states:
            hidden_states = self.get_hidden_states_multi_choice(outs, bt_size)
        for bt in range(bt_size):
            temp_res = {
                'Res': choices[max_indices[bt]],
                'Log_p':{
                    'token probs': next_token_probs[bt].tolist(),# choices prob
                    'token_entropy': float(entropy[bt]), # real entropy
                    'choices_entropy': float(choices_entropy[bt]) # probs = softmax(choices)
                },
            }
            if self.args.hidden_states:
                temp_res['hidden_states'] = hidden_states[bt]
            if self.args.output_states:
                temp_res['output_states'] = probs[bt]
            if self.args.attn_weights:
                temp_res['attn_weights'] = attentions[bt]
            self.outputs.append(temp_res)

    def calculate_res(self):
        all_data = self.data.data # 所有数据, 需要算结果的数据可能是其中一部分
        res = []
        begin = 0
        acc = 0
        print(f'len of all data: {len(all_data)}')
        for idx in range(len(all_data)):
            if idx not in self.data.idxs: # 不需要统计的数据
                res.append(all_data[idx])
            else:
                res_sample = {}
                if 'qa' in self.args.type:
                    res_sample['qa_prompt'] = self.data[begin]
                    res_sample['Res'] = self.outputs[begin]['Res']
                    res_sample['Log_p'] = self.outputs[begin]['Log_p']
                    if self.args.task == 'mmlu' or self.args.task == 'tq':
                        res_sample['question'] = self.data.format_example(all_data, idx, include_answer=False)
                        res_sample['has_answer'] = res_sample['Res'] == all_data[idx][-1]
                        res_sample['reference'] = all_data[idx][-1]
                        if self.args.attn_weights:
                            res_sample['attn_weights'] = self.outputs[begin]['attn_weights'].tolist()
                        if self.args.hidden_states:
                            res_sample['hidden_states'] = self.outputs[begin]['hidden_states']
                        if self.args.output_states:
                            res_sample['output_states'] = self.outputs[begin]['output_states'].tolist()
                    else:
                        res_sample['question'] = all_data[idx]['question']
                        res_sample['has_answer'] = has_answer(all_data[idx]['reference'], res_sample['Res'])
                        res_sample['reference'] = all_data[idx]['reference']
                    acc += res_sample['has_answer']
                res.append(res_sample)
                begin += 1
        print(f'processed data count: {begin}')
        print(f'accuracy: {acc / begin}')
        return res, acc / begin
    
    def get_hidden_states_multi_choice(self, outs, bt_size):
        """
        得到输出第一个token时每一层的hidden_state
        """
        res = [[] for _ in range(bt_size)]
        for layer in range(len(outs['hidden_states'][0])):
            item = outs['hidden_states'][0][layer] # bs, generated_len(input_len or 1), hidden_size
            for idx in range(bt_size):
                res[idx].append(item[idx][-1].tolist())
        return res
    
    
