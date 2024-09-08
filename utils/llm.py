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
        self.eos_id_dict = {
            'llama2-7b-chat': self.tokenizer.eos_token_id,
            'llama3-8b-instruct': self.tokenizer.convert_tokens_to_ids(['<|eot_id|>'])[0]
        }
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
                                       output_attentions=self.args.attn_weights, return_dict_in_generate=True, output_scores=True, output_hidden_states=self.args.hidden_states, 
                                       pad_token_id=0, top_p=1.0, temperature=1, do_sample=False)
            if self.args.task == 'mmlu' or self.args.task == 'tq':
                self.process_res_multi_choice(outs, input_ids) # 得到一个batch的结果
            else:
                self.process_res(outs, input_ids)
        print(f'len of outputs: {len(self.outputs)}')
        return self.calculate_res()
    
    def process_res(self, outs, inputs):
        """
        按batch处理模型generate输出, 得到输出文本,每个token的概率,以及每个token的entropy
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
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        input_len = inputs.shape[-1]
        bt_size = inputs.shape[0]
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        # print(f'text: {self.tokenizer.batch_decode(new_ids, skip_sepcial_tokens=True)}')
        end_idx = self.get_generation_end(new_ids)
        print(f'end_idx: {end_idx}')
        # 存储概率最大的token_id, 存储对应的probs, 存储seqs对应probs. 当且仅当使用greedy search时, top_indices=outs['sequence']
        top_indices, top_scores, ans_scores, ans_entropy = self.get_generated_tokens_probs_entropy(scores, new_ids, bt_size)

        if self.args.hidden_states:
            hidden_modes = self.args.hidden_idx_mode.split(',')
            all_modes_hidden_state = [{} for _ in range(bt_size)]
            for mode in hidden_modes:
                if mode == 'ans': #不支持提取answer部分第一个token的hidden state
                    raise ValueError('Do not support hidden_mode=ans for free-form qa')
                if mode == 'every': # 得到ans token在每一层的概率, 每一层的top-1 token
                    probs_for_generated_tokens, tokens_for_each_layer = self.get_token_and_prob_for_each_pos(outs, bt_size, end_idx) #(bt_size, layers, ans_len)
                else:
                    pos_idx = self.get_need_idx_for_generation(top_scores, end_idx, mode)
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, pos_idx, mode)
                    for bt in range(bt_size):
                        all_modes_hidden_state[bt][mode] = hidden_states[bt]

        for bt in range(bt_size):
            # print(f'ans: {self.tokenizer.decode(new_ids[bt][:end_idx[bt]])}')
            temp_res = ({
                'Res': self.tokenizer.decode(new_ids[bt][:end_idx[bt]]).strip(),
                'Log_p':{
                    'tokens': new_ids[bt][:end_idx[bt]].tolist(),
                    'token_probs': ans_scores[bt][:end_idx[bt]].tolist(),
                    'token_entropy': ans_entropy[bt][:end_idx[bt]].tolist()
                }
            })
            if self.args.hidden_states:
                if self.args.hidden_idx_mode == 'every':
                    temp_res['probs_for_generated_tokens'] = probs_for_generated_tokens[bt]
                    temp_res['tokens_for_each_layer'] = tokens_for_each_layer[bt]
                else:
                    temp_res['hidden_states'] = all_modes_hidden_state[bt]

            self.outputs.append(temp_res)

    def process_res_multi_choice(self, outs, inputs):
        """
        对多选问题,得到在输出的token上的结果,概率,entropy,hidden_state等信息
        Input:
            - outs:
            - inputs:
        Return:
            - 
        """
        choices = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'] # token可能有A和(A, 长度为8是为了对应
        input_len = inputs.shape[-1]
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        end_idx = self.get_generation_end(new_ids)
        # print(f'text: {self.tokenizer.batch_decode(new_ids, skip_sepcial_tokens=True)}')
        # print(f'end idx: {end_idx}')
        # 找到choice出现位置,以及对应的token id
        ans_token_idx, choices_idx = self.get_choice_idx(outs, inputs, end_idx)
        print(f'answer idx: {ans_token_idx}')
        need_scores = []
        bt_size = inputs.shape[0]
        for bt in range(bt_size):
            need_scores.append(scores[ans_token_idx[bt]][bt]) # vocab_size
        need_scores = torch.stack(need_scores)
        probs = nn.Softmax(dim=-1)(need_scores) # 词表中所有token概率
        next_token_probs = probs[:, choices_idx] # batch_size, 8
        entropy = torch.sum(-(probs * torch.log2(probs)), dim=-1) # batch_size, 8
        max_scores, max_indices = torch.max(next_token_probs, dim=-1) # 生成token
        # 得到所有token对应的prob,为提取min-prob token对应hidden state作准备
        _, top_scores, _, _ = self.get_generated_tokens_probs_entropy(scores, new_ids, bt_size)

        if self.args.attn_weights: 
            attentions = self.get_attn_multi_choice(outs, bt_size, ans_token_idx)

        if self.args.hidden_states:
            # 若有多种mode需要记录,则一次性记录所有mode的hidden state
            hidden_modes = self.args.hidden_idx_mode.split(',')
            all_modes_hidden_state = [{} for _ in range(bt_size)]
            for mode in hidden_modes:
                if mode == 'every': # 得到ans token在每一层的概率, 每一层的top-1 token
                    raise ValueError('Do not need to specify hidden_idx_mode=every for multi-choice qa')
                elif mode == 'ans': # 取response中ans的first token
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, ans_token_idx, mode)
                else:
                    pos_idx = self.get_need_idx_for_generation(top_scores, end_idx, mode)
                    hidden_states = self.get_hidden_states_for_given_pos(outs, bt_size, pos_idx, mode)
                for bt in range(bt_size):
                    all_modes_hidden_state[bt][mode] = hidden_states[bt]
            
        for bt in range(bt_size):
            temp_res = {
                'Res': choices[max_indices[bt]],
                'Log_p':{
                    'token probs': next_token_probs[bt].tolist(),# choices prob
                    'token_entropy': float(entropy[bt]), # real entropy
                },
                'end_idx': end_idx[bt]
            }
            if self.args.hidden_states:
                temp_res['hidden_states'] = all_modes_hidden_state[bt]
            if self.args.output_states:
                temp_res['output_states'] = probs[bt]
            if self.args.attn_weights:
                temp_res['attn_weights'] = attentions[bt]
            self.outputs.append(temp_res)

    def calculate_res(self):
        """
        保存输出结果
        """
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
                        res_sample['end_idx'] = self.outputs[begin]['end_idx']
                    else:
                        res_sample['question'] = all_data[idx]['question']
                        res_sample['has_answer'] = has_answer(all_data[idx]['reference'], res_sample['Res'])
                        res_sample['reference'] = all_data[idx]['reference']
                    if self.args.attn_weights:
                        res_sample['attn_weights'] = self.outputs[begin]['attn_weights'].tolist()
                    if self.args.hidden_states:
                        if self.args.hidden_idx_mode == 'every':
                            res_sample['probs_for_generated_tokens'] = self.outputs[begin]['probs_for_generated_tokens']
                            res_sample['tokens_for_each_layer'] = self.outputs[begin]['tokens_for_each_layer']
                        else:
                            res_sample['hidden_states'] = self.outputs[begin]['hidden_states']
                    if self.args.output_states:
                        res_sample['output_states'] = self.outputs[begin]['output_states'].tolist()
                    acc += res_sample['has_answer']
                res.append(res_sample)
                begin += 1
        print(f'processed data count: {begin}')
        print(f'accuracy: {acc / begin}')
        return res, acc / begin
    
    def get_hidden_states_for_given_pos(self, outs, bt_size, need_idx, mode='first'):
        """
        得到指定位置token生成时每一层的hidden_state
        Input:
            - out: generate结果
            - bt_size: batch size
            - need_idx: 每个batch生成结果中,需要获取hidden state的位置
            - need_layers: 需要获取的hidden states所在的层
        Return:
            - res: 每一层对应的hidden states, (batch_size, layers, hidden_dim)
        Note:
            - outs['hidden_states'] tuples of (genetared_token, layer)->(bs, generated_len, hidden_dim)
        """
        if self.args.need_layers == 'last':
            need_layers = [-1]
        elif self.args.need_layers == 'all':
            need_layers = range(len(outs['hidden_states'][0]))
        elif self.args.need_layers == 'mid':
            need_layers = [int(len(outs['hidden_states'][0]) / 2)]
        else:
            raise ValueError('Specify the wrong need_layers')
        
        res = [[] for _ in range(bt_size)]
        for bt in range(bt_size): # 遍历sample
            temp_idx = need_idx[bt] # 当前sample需要考虑的token的idx
            # print(f'need layers: {need_layers}')
            if type(temp_idx) != list: # 只需要取一个token
                for layer in need_layers: # 该token的每一层
                    hidden_states = outs['hidden_states'][temp_idx][layer][bt][-1] # bs, generated_len(input_len or 1), hidden_size
                    res[bt].append(hidden_states.to(torch.float16).tolist())
            else: # 取所有token
                for layer in need_layers: # 该token的每一层
                    temp_res = []
                    for item in temp_idx: # 所有需要考虑的tokens
                        temp_res.append(outs['hidden_states'][item][layer][bt][-1])
                    temp_res = torch.stack(temp_res)
                    if mode == 'avg':
                        res[bt].append(torch.mean(temp_res, dim=0).to(torch.float16).tolist())
                    elif mode == 'dim_min': # hidden state不同维度取min
                        res[bt].append(torch.min(temp_res, dim=0)[0].to(torch.float16).tolist())
                    elif mode == 'dim_max':
                        res[bt].append(torch.max(temp_res, dim=0)[0].to(torch.float16).tolist())
        return res

    def get_attn_multi_choice(self, outs, bt_size, need_idx):
        """
        提取选项生成时的各层attention weights
        Input:
            - out: generate结果
            - bt_size: batch size
            - need_idx: 每个batch生成结果中,选项token的idx
        Return:
            - res: 每一层中所有attn_head的注意力权重, (batch_size, layers, num_head, context_len)
        Note:
            - outs['attentions'] tuples of (genetared_token, layer)->(bs, num_head, generated_len, context_len)
        """
        res = [[] for _ in range(bt_size)]
        for bt in range(bt_size):
            temp_idx = need_idx[bt]
            for layer in range(len(outs['attentions'][temp_idx])): # temp_idx处token对应的所有层
                attentions = outs['attentions'][temp_idx][layer][bt, :, -1] # bs, head_num, seq_len(input_len)
                res[bt].append(attentions.tolist())
        return res

    def get_choice_idx(self, outs, inputs, end_idx):
        """
        找到每个样本中choice出现的位置
        """
        batch_size, input_len = inputs.shape
        choices = ['A', 'B', 'C', 'D', '(A)', '(B)', '(C)', '(D)']
        out_idx = [0 for _ in range(batch_size)] # 没找到就默认为第一个token
        seqs = outs['sequences'] # batch_size, seq_len, 存储的是token_id
        new_token_ids = seqs[:, input_len:]

        choices_idx = self.tokenizer(choices)['input_ids']
        if self.args.model_name == 'llama2-7b-chat':
            # ['<s>', '_A'],  ['<s>', '(', 'A', ')']
            choices_idx = [item[1] if len(item) == 2 else item[2] for item in choices_idx] # _A, A等的token_id
            #['_A'], ['(A', ')']
        elif self.args.model_name == 'llama3-8b-instruct':
            choices_idx = [item[0] for item in choices_idx]
        for bt in range(batch_size): # 遍历batch
            for idx in range(end_idx[bt]): # 一个序列中token
                token_id = new_token_ids[bt][idx]
                if token_id in choices_idx: # 第一个出现选项的位置
                    out_idx[bt] = idx
                    break
        return out_idx, choices_idx      

    def get_need_idx_for_generation(self, probs, end_idx, mode):
        """
        根据mode找到需要探测的token的index
        Input:
            - mode: 
                - first, last, min, avg - 得到需要的token的idx
                - dim_min, dim_max - 得到所有token的idx, 后续在hidden_dim上取min/max
        """ 
        res_idx = []
        bt_size = probs.shape[0]
        text_len = probs.shape[1]
        assert mode in ['first', 'last', 'avg', 'min', 'dim_min', 'dim_max']
        if mode == 'first':
            res_idx = torch.zeros(bt_size, dtype=torch.int) # 全选第一个位置
        elif mode == 'last':
            res_idx = [item if item != text_len else item - 1 for item in end_idx] # 全选最后一个位置
        elif mode == 'min':
            temp_idx = [item + 1 if item != text_len else item for item in end_idx]
            for bt in range(bt_size):
                min_prob, min_index = torch.min(probs[bt][:temp_idx[bt]], dim=-1) # batch_size
                res_idx.append(min_index)
        elif mode == 'avg' or mode == 'dim_min' or mode == 'dim_max':
            for bt in range(bt_size):
                if end_idx[bt] == text_len:
                    res_idx.append(list(range(end_idx[bt])))
                else:
                    res_idx.append(list(range(end_idx[bt] + 1)))
        return res_idx
    
    def get_token_and_prob_for_each_pos(self, outs, bt_size, end_idx):
        """
        得到每个位置每一层top-1 token(early exit), 最终生成的token在每一层的概率
        """
        probs_for_generated_token = [[] for _ in range(bt_size)] # 最终生成的token在每一层对应的概率
        tokens_for_each_pos = [[] for _ in range(bt_size)] #
        for bt in range(bt_size):
            end_pos = end_idx[bt]
            for pos in range(end_pos):
                hidden_states_for_all_layers = []
                for layer in range(len(outs['hidden_states'][pos]))[1:]:
                    hidden_states = outs['hidden_states'][pos][layer][bt][-1] # hidden_size
                    hidden_states_for_all_layers.append(hidden_states)
                hidden_states_for_all_layers = torch.stack(hidden_states_for_all_layers) # (layers, hidden_dim)
                probs = nn.Softmax(dim=-1)(self.model.lm_head(hidden_states_for_all_layers))
                max_value_for_each_layer, max_token_for_each_layer = torch.max(probs, dim=-1)
                tokens_for_each_pos[bt].append(self.tokenizer.convert_ids_to_tokens(max_token_for_each_layer))
                generated_token = max_token_for_each_layer[-1]
                probs_for_generated_token[bt].append(probs[:, generated_token])
            
            probs_for_generated_token[bt] = torch.stack(probs_for_generated_token[bt]).t().tolist()
            probs_for_generated_token[bt] = [[round(element, 4) for element in row] for row in probs_for_generated_token[bt]]
            tokens_for_each_pos[bt] = [[tokens_for_each_pos[bt][j][i] for j in range(len(tokens_for_each_pos[bt]))] for i in range(len(tokens_for_each_pos[bt][0]))]
        return probs_for_generated_token, tokens_for_each_pos
    
    def get_generation_end(self, generated_tokens):
        # generated_tokens batch_size, new_seq_len
        text_len = generated_tokens.shape[-1]
        end_idx = []
        for idx in range(len(generated_tokens)):
            eos_idx = torch.where(generated_tokens[idx] == self.eos_id_dict[self.args.model_name])[0] # 返回tuple, [0]是该元素出现位置的tensor
            if len(eos_idx) == 0: # 没有eos_token
                end_idx.append(text_len)
            else:
                end_idx.append(eos_idx[0].item()) # eos_token出现的第一个位置
        return end_idx
    
    def get_generated_tokens_probs_entropy(self, scores, generated_tokens, bt_size):
        top_indices = [] # 存储概率最大的token_id
        top_scores = [] # 存储对应的probs
        ans_scores = [] # 存储seqs对应probs
        ans_entropy = []
        for idx in range(len(scores)): # 遍历每个token
            probs = nn.Softmax(dim=1)(scores[idx]) # batch_size, vocab_size
            tmp_scores, tmp_indices = torch.max(probs, dim=1) # batch_size
            cur_scores = [probs[t, generated_tokens[t, idx]] for t in range(bt_size)] # batch_size, 每个生成token的概率
            cur_entropy = torch.sum(-(probs * torch.log2(probs)), dim=1) # batch_size

            # 当且仅当使用greedy search时, ans_scores = top_scores
            ans_scores.append(cur_scores) # seq_len, batch_size
            ans_entropy.append(cur_entropy.tolist())
            top_indices.append(tmp_indices.tolist())
            top_scores.append(tmp_scores.tolist())
        
        top_indices = torch.tensor(top_indices, dtype=torch.int64).t()
        top_scores = torch.tensor(top_scores).t() # batch_size, text_len
        ans_scores = torch.tensor(ans_scores).t()
        ans_entropy = torch.tensor(ans_entropy).t()
        return top_indices, top_scores, ans_scores, ans_entropy