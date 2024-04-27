import json
import jsonlines
import math
import numpy as np
import string
import scipy.stats
from utils.utils import has_answer, remove_punc, remove_stopwords

def get_spearman_coefficient(x, y):
    assert len(x) == len(y)
    return scipy.stats.spearmanr(x, y)[0]

def adaptive_retrieval(model_data, ra_data):
    score_list, c_score_list, unc_score_list = [], [], []

    become_right = []
    become_wrong = []
    for idx in range(len(model_data)):
        if 'info' in model_data[idx]:
            continue
        if model_data[idx]['Giveup'] == True:
            score_list.append(ra_data[idx]['has_answer'])
            unc_score_list.append(ra_data[idx]['has_answer'])
            if model_data[idx]['has_answer'] == 0 and ra_data[idx]['has_answer'] == 1:
                become_right.append(idx)
            if model_data[idx]['has_answer'] == 1 and ra_data[idx]['has_answer'] == 0:
                become_wrong.append(idx)
        else:
            score_list.append(model_data[idx]['has_answer'])
            c_score_list.append(ra_data[idx]['has_answer'])
    print(f'count: {len(score_list)}')
    print(f'has_answer: {sum(score_list) / len(score_list)}')
    print(f'uncertain ra has answer: {sum(unc_score_list) / len(unc_score_list)}')
    print(f'certain ra has answer: {sum(c_score_list) / len(c_score_list)}')
    print(f'become right: {len(become_right) / len(unc_score_list)}')
    print(f'become wrong: {len(become_wrong) / len(unc_score_list)}')

def compute_score(data, origin_data=[]):
    score_list = []
    em_list = []
    f_list = []
    become_wrong = []
    become_right = []
    for idx in range(len(data)):
        sample = data[idx]
        if 'has_answer' not in sample:
            continue
        if origin_data != []:
            if origin_data[idx]['has_answer'] == 0 and sample['has_answer'] == 1:
                become_right.append(idx)
            if origin_data[idx]['has_answer'] == 1 and sample['has_answer'] == 0:
                become_wrong.append(idx)
        score_list.append(sample['has_answer'])
        em_list.append(sample['EM'])
        f_list.append(sample['F1'])
    print(f'count: {len(em_list)}')
    print(f'em: {sum(em_list) / len(em_list)}')
    print(f'f1: {sum(f_list) / len(f_list)}')
    print(f'has answer: {sum(score_list) / len(score_list)}')
    print(f'become right: {len(become_right) / len(score_list)}') 
    print(f'become wrong: {len(become_wrong) / len(score_list)}')
                              
def compute_doc_p(data, key):
    p = []
    for sample in data:
        if 'info' in sample:
            continue
        p.append(has_answer(sample['reference'], sample[key][0]))
    print(key)
    print(f'precision@1: {sum(p) / len(p)}')

def compute_giveup_score(data, sample_data=[]):
    """
    计算data中的Giveuo指标以及分数指标
    - compare_data, 与idx_list起到相同作用,在对比时过滤不合规数据
    """
    giveup_list, score_list, c_score_list, unc_score_list, align = [], [], [], [], []
    cnt = 0
    post_cnt = 0
    for idx in range(len(data)):
        sample = data[idx]
        if 'idx' in sample:
            cnt += 1
        if 'confidence_replace' in sample:
            post_cnt += 1
        if 'has_answer' not in sample:
            continue
        score_list.append(sample['has_answer'])
        if sample['has_answer'] != sample['Giveup']:
            align.append(1)

        if sample['Giveup'] == True:
            unc_score_list.append(sample['has_answer'])
        else:
            c_score_list.append(sample['has_answer'])
        giveup_list.append(sample['Giveup'])
    print(f'conut: {len(giveup_list)}')
    print(f'count of non-post data: {cnt}')
    print(f'count of post data: {post_cnt}')
    print(f'uncertain ratio: {sum(giveup_list) / len(giveup_list)}')
    print(f'has answer: {sum(score_list) / len(score_list)}')
    print(f'uncertain has answer: {sum(unc_score_list) / len(unc_score_list)}')
    print(f'certain has answer: {sum(c_score_list) / len(c_score_list)}')
    compute_overconfidence([sum(giveup_list) / len(giveup_list)], [sum(c_score_list) / len(c_score_list)])
    compute_conservation([sum(giveup_list) / len(giveup_list)], [sum(unc_score_list) / len(unc_score_list)])
    print(f'alignment: {sum(align) / len(giveup_list)}')

def compute_overconfidence(unc_list, c_acc):
    assert len(unc_list) == len(c_acc)
    res = []
    for idx in range(len(unc_list)):
        overconf = (1-unc_list[idx]) * (1-c_acc[idx])
        res.append(overconf)
        print(f"overconf: {format(overconf, '.4f')}")
    # print(res)
def compute_conservation(unc_list, unc_acc):
    assert len(unc_list) == len(unc_acc)
    res = []
    for idx in range(len(unc_list)):
        conserv = (unc_list[idx]) * (unc_acc[idx])
        res.append(conserv)
        print(f"conserv: {format(conserv, '.4f')}")

def get_giveup_after_challenge(data, challenge_data):
    """
    data: 包含检索增强前后信息的数据
    challenge_data: 进行challenge之后的数据
    """
    for idx in range(len(data)):
        sample = data[idx]
        data[idx]['challenge'] = challenge_data[sample['nq_idx']]['Giveup']
    return data

def change_giveup_after_challenge(data, challenge_data):
    for idx in range(len(data)):
        data[idx]['Giveup'] = challenge_data[idx]['Giveup']
    return data


def get_data_before_and_after_ra(qa_data, ra_data, replace_data, origin_data, origin_replace_data, ctx_wrong='right'):
    """
    Input:
        qa_data: nq数据集
        ra_data: 检索增强后得到的结果
        origin_data: 没有经过检索增强得到的结果
    Output: 
        res_data: 包含增强前后的数据
        res_idx: 同时包含在davinci和chatgpt里的数据的idx
    """
    giveup_ratio = []
    res_data = []
    has_tom = []
    ctx_dict = {'wrong': 'dpr_ctx_wrong', 'right': 'dpr_ctx'}
    data = ra_data
    ctx_key = ctx_dict[ctx_wrong]
    res_idx = []
    for idx in range(len(ra_data)):
        # 过滤ra中不合规数据
        if 'Res' not in ra_data[idx]:
            continue
        
        if len(data[idx]['Res']) <= 1:
            data[idx]['Res'] = replace_data[idx]['Prediction']
        if len(origin_data[idx]['Res']) <= 1:
            origin_data[idx]['Res'] = origin_replace_data[idx]['Prediction']
        if 'tom' in data[idx]['Res'].lower():
            has_tom.append(1)
        res_idx.append(idx)
        res_data.append({'question': qa_data[idx]['question'], 
                         'dpr_ctx': qa_data[idx][ctx_key], # 正确还是错误文档
                         'ans': qa_data[idx]['reference'], 
                         'pred': data[idx]['Res'],
                         'origin_pred': origin_data[idx]['Res'],
                         'EM': data[idx]['EM'],
                         'F1': data[idx]['F1'],
                         'Giveup': data[idx]['Giveup'],
                         'EM_origin': origin_data[idx]['EM'],
                         'F1_origin': origin_data[idx]['F1'],
                         'Giveup_origin': origin_data[idx]['Giveup'],
                         'nq_idx': idx # 在原始nq数据中的idx
                         })
        giveup_ratio.append(data[idx]['Giveup'])
    print(f'count: {len(giveup_ratio)}')
    return res_data, res_idx

def compute_score_before_and_after_ra(data, idx_list=[]):
    """
    计算检索增强前后的相关分数
    Input:
        idx_list, 用于过滤一些数据(在对比时)
    Output:
        new_data: 过滤后, 区分Giveup的数据
    """
    new_data = []
    res_giveup = []
    res_origin_giveup = []
    align = []
    align_before = []
    for idx in range(len(data)):
        if len(idx_list) != 0:
            if data[idx]['nq_idx'] not in idx_list:
                continue
        if data[idx]['Giveup_origin'] != False: # 仅考虑模型有信心/没有信心的数据
            continue
        if has_answer(data[idx]['ans'], data[idx]['pred']) != data[idx]['Giveup']:
            align.append(1)
        if has_answer(data[idx]['ans'], data[idx]['origin_pred']) != data[idx]['Giveup_origin']:
            align_before.append(1)
        res_giveup.append(data[idx]['Giveup'])
        res_origin_giveup.append(data[idx]['Giveup_origin'])
        new_data.append(data[idx])
    print(f'count: {len(res_giveup)}')
    print(f'uncertain ratio: {sum(res_giveup) / len(res_giveup)}')
    print(f'uncertain ratio origin: {sum(res_origin_giveup) / len(res_origin_giveup)}')
    print(f'alignment: {len(align) / len(res_giveup)}')
    print(f'alignment origin: {len(align_before) / len(res_giveup)}')
    return new_data

def pred_term_in_doc(sample, right_wrong):
    origin_pred = remove_stopwords(remove_punc(sample['origin_pred'].lower()).split())
    pred = remove_stopwords(remove_punc(sample['pred'].lower()).split())
    question = sample['question'].lower().split()
    # 过滤无法区别是否依赖文档的数据
    if right_wrong == 'right':
        if has_answer(sample['ans'], sample['origin_pred']): 
            if pred == origin_pred:
                raise ValueError('not sure')
    clean_pred = [item for item in pred if item not in question]
    clean_origin = [item for item in origin_pred if item not in question]

    clean_score = []
    origin_score = []
    doc = sample['dpr_ctx'][0].lower().split()
 
    for t in clean_pred: 
        add_score = 1 if t in doc else 0
        clean_score.append(add_score)

    for t in clean_origin:
        add_score = 1 if t in doc else 0
        origin_score.append(add_score)

    return (sum(clean_score) / len(clean_score)) > (sum(origin_score) / len(origin_score))


def answer_change_ratio(data, right_wrong):
    """
    计算检索增强前后各种指标的变化, 与计算分数的函数不同,这里主要关注不同情况下的变化/对比
    data: get_score输出格式的数据, 包含ra前后数据
    """
    data_cnt = 0
    res_data = []
    cnt3 = 0
    cnt_has_answer_change = []
    cnt_has_answer_change_0 = []
    cnt_has_answer_change_1 = []
    cnt_has_answer = 0
    cnt_has_answer_before = 0
    cnt_answer_change = 0
    cnt_pred_in_doc = []
    ans_in_doc = 0
    uncertain = 0
    uncertain_before = 0
    cnt_pred_in_doc_1 = []
    cnt_pred_in_doc_0 = []
    align = []
    align_before = []
    for idx in range(len(data)):
        sample = data[idx]    
        data_cnt += 1
        if has_answer(sample['ans'], sample['origin_pred']) != sample['Giveup_origin']:
            align_before.append(1)
        
        if has_answer(sample['ans'], sample['pred']) != sample['Giveup']:
            align.append(1)

        if has_answer(sample['ans'], sample['origin_pred']):# 原来是对的
            cnt_has_answer_change_1.append(has_answer(sample['ans'], sample['origin_pred']) != has_answer(sample['ans'], sample['pred']))
            try:
                cnt_pred_in_doc_1.append(pred_term_in_doc(sample, right_wrong))
            except:
                pass

        if has_answer(sample['ans'], sample['origin_pred']) == False: #原来是错的
            cnt_has_answer_change_0.append(has_answer(sample['ans'], sample['origin_pred']) != has_answer(sample['ans'], sample['pred']))
            try:
                cnt_pred_in_doc_0.append(pred_term_in_doc(sample, right_wrong))
            except:
                pass
        if 'tom' in sample['pred'].lower():
            cnt3 += 1
        
        if has_answer(sample['ans'], sample['origin_pred']) == False: # 之前不对, 现在对了 
            if has_answer(sample['ans'], sample['pred']): 
                res_data.append(1)
            else:
                res_data.append(0)

        cnt_has_answer_change.append(has_answer(sample['ans'], sample['origin_pred']) != has_answer(sample['ans'], sample['pred']))
        if has_answer(sample['ans'], sample['origin_pred']):
            cnt_has_answer_before += 1
        if has_answer(sample['ans'], sample['pred']):
            cnt_has_answer += 1
        if has_answer([sample['origin_pred']], sample['pred']) == False: # answer变化
            cnt_answer_change += 1
            
        try:
            cnt_pred_in_doc.append(pred_term_in_doc(sample, right_wrong))
        except:
            pass
        
        if has_answer(sample['ans'], sample['dpr_ctx'][0]):
            ans_in_doc += 1
        uncertain += sample['Giveup']
        uncertain_before += sample['Giveup_origin']
    print(f'count: {data_cnt}')
    print(f'cnt pred in doc - term level: {sum(cnt_pred_in_doc) / len(cnt_pred_in_doc)}')
    print(f'0: {sum(cnt_pred_in_doc_0) / len(cnt_pred_in_doc_0)}')
    print(f'1: {sum(cnt_pred_in_doc_1) / len(cnt_pred_in_doc_1)}')
    print(f'has answer change: {sum(cnt_has_answer_change) / len(cnt_has_answer_change)}')
    print(f'has answer change 0: {sum(cnt_has_answer_change_0) / len(cnt_has_answer_change_0)}')
    print(f'has answer change 1: {sum(cnt_has_answer_change_1) / len(cnt_has_answer_change_1)}')
    print(f'has tom: {cnt3 / data_cnt}')
    print(f'has answer: {cnt_has_answer / data_cnt}')
    print(f'has answer before: {cnt_has_answer_before / data_cnt}')
    print(f'ans in doc: {ans_in_doc}')
    print(f'uncertain: {uncertain / data_cnt}')
    print(f'uncertain before: {uncertain_before / data_cnt}')
    print(f'alignment: {len(align) / data_cnt}')
    print(f'alignment before: {len(align_before) / data_cnt}')

    return res_data

def get_answer_tokens(sample):
    """
    计算除了pattern,标点符号以外的tokens的个数(answer的token长度)
    """
    punc_idx = []
    new_tokens = [item.strip() for item in sample['Log_p']['tokens']]
    # 去除标点
    for i in range(len(new_tokens)):
        if new_tokens[i] in string.punctuation:
            punc_idx.append(i)
    # 去除uncertain等词
    if 'idx' in sample:
        for i in sample['idx']:
            punc_idx.append(i)
    return len(new_tokens) - len(punc_idx)

def compute_ppl(sample):
    """
    计算给定样本(sample)中答案的ppl
    """
    punc_idx = []
    print(sample['Log_p']['tokens'])
    new_tokens = [item.strip() for item in sample['Log_p']['tokens']]
    # 去除标点
    for i in range(len(new_tokens)):
        if new_tokens[i] in string.punctuation:
            punc_idx.append(i)
    # 去除uncertain等词
    if 'idx' in sample:
        for i in sample['idx']:
            punc_idx.append(i)

    ppl = 0
    cnt = 0
    for idx in range(len(sample['Log_p']['token_logprobs'])):
        if idx not in punc_idx:
            ppl += -sample['Log_p']['token_logprobs'][idx]
            cnt += 1
    if cnt == 0:
        return 0
    return ppl / cnt

def get_confidence_ppl(sample, dir=False):
    p = 0
    cnt = 0
    if dir == False:
        if 'idx' in sample:
            for idx in sample['idx']:
                p += -sample['Log_p']['token_logprobs'][idx]
                cnt += 1
            return p / cnt
        raise ValueError('There is no "idx" in the sample')
    else:
        p = compute_ppl(sample)
        return p

def get_entropy_form_dict(log_dict):
    res = 0
    for k, log_p in log_dict.items():
        res += -math.exp(log_p) * log_p
    return res

def get_entropy(sample):
    punc_idx = []
    new_tokens = [item.strip() for item in sample['Log_p']['tokens']]
    # 去除标点
    for i in range(len(new_tokens)):
        if new_tokens[i] in string.punctuation:
            punc_idx.append(i)
    # 去除uncertain等词
    if 'idx' in sample:
        for i in sample['idx']:
            punc_idx.append(i)

    entropy = 0
    cnt = 0
    for idx in range(len(sample['Log_p']['top_logprobs'])):
        if idx not in punc_idx:
            entropy += get_entropy_form_dict(sample['Log_p']['top_logprobs'][idx])
            cnt += 1
    if cnt == 0:
        return 0
    return entropy / cnt

def get_confidence_entropy(sample, dir):
    entropy = 0
    cnt = 0
    if dir == False:
        if 'idx' in sample:
            for idx in sample['idx']:
                entropy += get_entropy_form_dict(sample['Log_p']['top_logprobs'][idx])
                cnt += 1
            return entropy / cnt
        raise ValueError('There is no "idx" in the sample')
    else:
        entropy = get_entropy(sample)
        return entropy

def compute_p(sample):
    punc_idx = []
    new_tokens = [item.strip() for item in sample['Log_p']['tokens']]
    # 去除标点
    for i in range(len(new_tokens)):
        if new_tokens[i] in string.punctuation:
            punc_idx.append(i)
    # 去除uncertain等词
    if 'idx' in sample:
        for i in sample['idx']:
            punc_idx.append(i)

    p = 0
    cnt = 0
    for idx in range(len(sample['Log_p']['token_logprobs'])):
        if idx not in punc_idx:
            p += math.exp(sample['Log_p']['token_logprobs'][idx])
            cnt += 1
    return p / cnt
    