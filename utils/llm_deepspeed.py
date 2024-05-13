import time
import os
from .utils import deal_answer, deal_judge, deal_post, str2paras, deal_judge_new, has_answer
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
    
class ParallelGenerater:
    def __init__(self, args):
        self.args = args
        self.model = LlamaForCausalLM.from_pretrained(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.batch_size = args.batch_size
        self.outputs = []
        print('load generater finish.')
        self.deepspeed_model()
        print(f'deepspeed initialize.')

    def deepspeed_model(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
        # distributed setup
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
        config = AutoConfig.from_pretrained(self.args.model_path)
        model_hidden_size = config.hidden_size
        # batch size has to be divisible by world_size, but can be bigger than world_size
        train_batch_size = 1 * world_size
        ds_config = {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": model_hidden_size * model_hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
                "stage3_param_persistence_threshold": 10 * model_hidden_size
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }
        dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
        self.ds_engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
        self.ds_engine.module.eval()  # inference

    def load_data(self, data):
        self.data = data
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=self.batch_size)
    
    def process_res(self, outs, inputs):
        # attention和scores都不包含输入
        attentions = outs['attentions'] # tuple(generated_token, layer) -> (batch_size, num_heads, generated_length, sequence_length)
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        seqs = outs['sequences'] # batch_size, seq_len
        input_len = inputs.shape[-1]
        bt_size = inputs.shape[0]
        new_ids = seqs[:, input_len:] # batch_size, new_seq_len
        # print(f'text: {self.tokenizer.batch_decode(new_ids, skip_sepcial_tokens=True)}')
        # 得到每个seq真正的结尾
        text_len = new_ids.shape[-1]
        end_idx = []
        for idx in range(len(new_ids)):
            eos_idx = torch.where(new_ids[idx] == self.tokenizer.eos_token_id)[0] # 返回tuple, [0]是该元素出现位置的tensor
            if len(eos_idx) == 0:
                end_idx.append(text_len)
            else:
                end_idx.append(eos_idx[0]) # 第一个位置
        top_indices = [] # 存储概率最大的token_id
        top_scores = [] # 存储对应的probs
        ans_scores = [] # 存储seqs对应probs
        ans_entropy = []
        for idx in range(len(scores)):
            probs = nn.Softmax(dim=1)(scores[idx]) # batch_size, vocab_size
            cur_scores = [probs[t, new_ids[t, idx]] for t in range(bt_size)] # batch_size 
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
        choices = ['A', 'B', 'C', 'D']
        # get attn_weights when generating the first token
        attentions = outs['attentions'][0][-1][:, :, -1] # bs, head_num, seq_len(input_len)
        choices_idx = self.tokenizer(choices)['input_ids']
        choices_idx = [item[1] for item in choices_idx] # <s> when idx=0
        scores = outs['scores'] # tuple of tensor (generated_len) -> (batch_size, vocab_size)
        bt_size = inputs.shape[0]
        choices_probs = nn.Softmax(dim=1)(scores[0][:, choices_idx])
        probs = nn.Softmax(dim=1)(scores[0])
        next_token_probs = probs[:, choices_idx] # batch_size, 4
        entropy = torch.sum(-(probs * torch.log2(probs)), dim=1) # batch_size
        choices_entropy = torch.sum(-(choices_probs * torch.log2(choices_probs)), dim=1)
        # print(f'next token probs: {next_token_probs}')
        max_scores, max_indices = torch.max(next_token_probs, dim=1)
        hidden_states = self.get_hidden_states_multi_choice(outs, bt_size)
        for bt in range(bt_size):
            # print(f'ans: {choices[max_indices[bt]]}')
            self.outputs.append({
                'Res': choices[max_indices[bt]],
                'Log_p':{
                    'token probs': next_token_probs[bt].tolist(),# choices prob
                    'token_entropy': float(entropy[bt]), # real entropy
                    'choices_entropy': float(choices_entropy[bt]) # probs = softmax(choices)
                },
                'attn_weights': attentions[bt],
                'hidden_states': hidden_states[bt]
            })
            
    def get_res(self):
        self.outputs = []
        device = torch.device('cuda')
        print(f'model device: {self.ds_engine.module.device}')
        self.device = device
        self.model.to(device)
        for batch in tqdm(self.dataloader):
            batch = self.tokenizer(batch, return_tensors='pt', padding=True).to(device)
            input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
            outs = self.ds_engine.module.generate(input_ids, attention_mask=attn_mask, max_new_tokens=self.args.max_new_tokens, output_attentions=True, return_dict_in_generate=True, output_scores=True, output_logits=True, output_hidden_states=True, pad_token_id=0, top_p=1.0, temperature=1, synced_gpus=True)
            if self.args.task == 'mmlu':
                self.process_res_multi_choice(outs, input_ids) # 得到一个batch的结果
            else:
                self.process_res(outs, input_ids)
        print(f'len of outputs: {len(self.outputs)}')
        return self.calculate_res()

    def calculate_res(self):
        all_data = self.data.data # 所有数据, 需要算结果的数据可能是其中一部分
        res = []
        begin = 0
        acc = 0
        print(f'len of all data: {len(all_data)}')
        for idx in range(len(all_data)):
            if idx not in self.data.idxs:
                res.append(all_data[idx])
            else:
                res_sample = {}
                if self.args.type == 'qa':
                    res_sample['qa_prompt'] = self.data[begin]
                    res_sample['Res'] = self.outputs[begin]['Res']
                    res_sample['Log_p'] = self.outputs[begin]['Log_p']
                    if self.args.task == 'mmlu':
                        res_sample['question'] = self.data.format_example(all_data, idx, include_answer=False)
                        res_sample['has_answer'] = res_sample['Res'] == all_data[idx][-1]
                        res_sample['reference'] = all_data[idx][-1]
                        res_sample['attn_weights'] = self.outputs[begin]['attn_weights'].tolist()
                        res_sample['hidden_states'] = self.outputs[begin]['hidden_states']
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
        res = [[] for _ in range(bt_size)]
        for layer in range(len(outs['hidden_states'][0])):
            item = outs['hidden_states'][0][layer] # bs, generated_len, hidden_size
            for idx in range(bt_size):
                res[idx].append(item[idx][-1].tolist())
        return res
