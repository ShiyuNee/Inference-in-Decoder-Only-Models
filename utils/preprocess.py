import json
import jsonlines
import math
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import deal_answer, has_answer, deal_judge_new
from utils.plot import read_json, write_jsonl
from utils.compute import compute_ppl
import string
from matplotlib.pyplot import MultipleLocator

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

def remove_pattern(text, patterns):
    text = text.lower()
    for item in patterns:
        text = text.replace(item, '')
    return text

def get_pattern_idx(path, pattern):
    """
    记录pattern在tokens中的idx
    """
    cnt = 0
    origin_data = read_json(path)
    max_len = [2, 1, 0] #最大token序列长度
    # 寻找certain/uncertain对应的索引
    for idx in range(len(origin_data)):
        if 'Res' not in origin_data[idx] or origin_data[idx]['Res'] == None:
            continue
        if 'idx' in origin_data[idx]:
            origin_data[idx].pop('idx')
        flag = 0 # 是否匹配上 pattern
        new_tokens = [i.strip() for i in origin_data[idx]['Log_p']['tokens']]
        end_idx = len(new_tokens) - 1 # 从后往前先找是否有uncertain, 再找是否有certain
        while end_idx >= 0:
            for tok_num in max_len:
                start_idx = end_idx - tok_num
                if start_idx >= 0:
                    cur_text = ''.join(new_tokens[start_idx: end_idx + 1])
                    if cur_text.lower() in pattern:
                        origin_data[idx]['idx'] = list(range(start_idx, end_idx + 1))
                        flag = 1
                        cnt += 1
                        break
            end_idx -= 1 # 结尾从后往前退
            if flag == 1:
                break
        # if flag == 0:
        #     print(origin_data[idx]['Res'])
    print(f'match pattern: {cnt}')
    print(f'all count: {len(origin_data)}')
    write_jsonl(origin_data, path) # 更新当前文件
    
def change_file(path, out_path, replace_path, qa_path, ref, mode, post_path="", confidence_idx_path="", replace_idx_path=""):
    """
    替换answer中的uncertain为空
    计算EM, F1, has_answer分数(答案长度<=1就用replace_data计算分数)
    对davinci调用get_pattern_idx,获得pattern在tokens中对应的idx, 对chatgpt添加伪idx(方便后续计算)
    - ref: 数据集中reference对应的key
    - mode: davinci/chatgpt
    """
    pattern = ['uncertainty', 'certainty', 'uncertain', 'certainly', 'certain', 'unsure']
    idx_list, cnt_list = [], []
    data = read_json(path)
    qa_data = read_json(qa_path)
    replace_data = read_json(replace_path) if replace_path != '' else []
    post_data = read_json(post_path) if post_path != "" else []
    no_answer = 0
    # 替换答案中在pattern中存在的字符串
    for idx in range(len(data)):
        data[idx]['question'] = qa_data[idx]['question']
        if 'dpr_ctx_wrong' in qa_data[idx]: # 带有dpr_doc时,保存相关信息方便post处理
            data[idx]['dpr_ctx'] = qa_data[idx]['dpr_ctx']
            data[idx]['dpr_ctx_wrong'] = qa_data[idx]['dpr_ctx_wrong']

        if 'Res' not in data[idx] or data[idx]['Res'] == None: # 过滤不合规数据
            continue
        new_res = remove_pattern(data[idx]['Res'], pattern).strip()
        save_res = new_res
        if new_res != data[idx]['Res'].lower(): # 存在pattern
            if mode == 'chat': # 为chat添加假的idx, 因为chat模型不提供token信息
                data[idx]['idx'] = [-1]
        else: # 不存在,需要post
            cnt_list.append(idx)
            if len(post_data) != 0:
                data[idx]['Giveup'] = post_data[idx]['Giveup']
                data[idx]['confidence_replace'] = 1 # 代表该数据的confidence是post_data中得到的
        # 算分数
        if len(new_res) <= 1:
            if confidence_idx_path == "":
                new_res = replace_data[idx]['Prediction']
            no_answer += 1
            idx_list.append(idx) # replace_idx
        em, f_temp = deal_answer(new_res, qa_data[idx][ref])
        has_temp = has_answer(qa_data[idx][ref], new_res)
        data[idx]['EM'] = em
        data[idx]['F1'] = f_temp
        data[idx]['has_answer'] = has_temp
        data[idx]['Res'] = save_res # 保存的是原始答案去除pattern, 不是replace answer
    print(f'pattern no match count: {len(cnt_list)}')
    print(f'replace count: {no_answer}')

    write_jsonl(data, out_path)
    if mode == 'davinci':
        get_pattern_idx(out_path, pattern)
    if confidence_idx_path != "":
        write_jsonl(cnt_list, confidence_idx_path) # 保存需要post处理的数据idx
        write_jsonl(idx_list, replace_idx_path) # 记下需要replace data的数据idx
def save_ppl(giveup_ppl):
    """
    利用计算EM的答案(长度<=1的全替换),计算ppl并保存在原文件中
    """
    data = read_json(giveup_ppl)
    replace_data = read_json('./data/source/test_res/text-davinci-003/nq_origin_qa_res.jsonl')
    for idx in range(len(data)):
        item = data[idx]
        if len(item['Res']) <= 1:
            item = replace_data[idx]
        data[idx]['ppl'] = compute_ppl(item)
    write_jsonl(data, giveup_ppl)

def merge_post_files(qa_path, post_path):
    qa_data = read_json(qa_path)
    post_data = read_json(post_path)
    assert len(qa_data) == len(post_data)
    for idx in range(len(qa_data)):
        qa_data[idx]['Giveup'] = deal_judge_new(post_data[idx]['Res'])
    return qa_data

if __name__ == '__main__':
    path = './data/source/test_res/text-davinci-003/nq_prompt7_giveup_res_ra.jsonl'
    out_path = './data/source/test_res/text-davinci-003/nq_prompt7_giveup_res_ra_new.jsonl'
    change_file(path, out_path)