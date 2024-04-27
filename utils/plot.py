import json
import jsonlines
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from utils.utils import *
from utils.compute import *
import string
from matplotlib.pyplot import MultipleLocator

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def scatter_density_plot(x, y, res_path):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy) # 估计每个点附近的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots()
    plt.scatter(x, y, c=z,  s=10, cmap='Spectral')
    plt.colorbar()
    plt.savefig(res_path)

def get_filter_data(giveup_data, score_data):
    new_score_data = []
    new_replace_data = []
    new_giveup_data = []
    replace_data = read_json('./data/source/test_res/text-davinci-003/nq_origin_qa_res.jsonl')
    for idx in range(len(score_data)):
        if score_data[idx]['Giveup'] == False and score_data[idx]['EM'] == 0:
            continue
        new_score_data.append(score_data[idx])
        new_replace_data.append(replace_data[idx])
        new_giveup_data.append(giveup_data[idx])
    sample_dis_plot(new_giveup_data, new_score_data, new_replace_data, 'prompt7_filter', False)
            
def get_sorted_ppl_list(score_data, replace_data, giveup_data, origin, skip=False):
    """
    对给定的数据,计算符合条件(条件在函数中自定义)的所有数据的ppl_list, sorted_ppl, no_answer
    no_answer: 只回答giveup,没给答案的数据列表,如果origin=true,则这个列表为空
    """
    ppl_list = []
    no_answer = []
    new_score_data = []
    new_giveup_data = [] 
    for idx in range(len(score_data)):
        if 'Res' not in giveup_data[idx]:
            continue
        # if skip and len(giveup_data[idx]['Res']) <= 1: # 排除只回答pattern,没给答案的样本
        #     continue
        new_score_data.append(score_data[idx])
        new_giveup_data.append(giveup_data[idx])
        if origin == False:
            if len(score_data[idx]['Res']) <= 1:
                no_answer.append(score_data[idx]['Giveup'] == True)
                score_data[idx] = replace_data[idx] # 将没有答案的样本, 更换为强制回答的答案(无论是否giveup)
            else:
                no_answer.append(0)
        temp_ppl = compute_ppl(score_data[idx])
        ppl_list.append(temp_ppl)
    sorted_ppl = sorted(enumerate(ppl_list), key=lambda x: x[1])
    return ppl_list, sorted_ppl, no_answer, new_score_data, new_giveup_data

def get_idx_for_each_section(sorted_ppl, ppl_dis):
    # 得到每个ppl区间对应的数据原始idx
    idx = 0
    item_idx = 0
    idx_list = []
    temp_ppl = []
    plot_x = []
    plot_count = []
    while item_idx < len(sorted_ppl):
        item = sorted_ppl[item_idx]
        if idx < len(ppl_dis) - 1:
            if item[1] <= ppl_dis[idx + 1]:
                temp_ppl.append(item[0])
                item_idx += 1
            else:
                if len(temp_ppl) != 0:
                    plot_x.append(str(ppl_dis[idx]))
                    plot_count.append(len(temp_ppl))
                    idx_list.append(temp_ppl)
                    temp_ppl = []
                idx += 1
        else:
            temp_ppl.append(item[0])
            item_idx += 1
    # 得到按ppl分后的index列表
    if len(temp_ppl) != 0:
        plot_x.append(str(ppl_dis[-1]))
        plot_count.append(len(temp_ppl))
        idx_list.append(temp_ppl)
    print(sum(plot_count))
    print(f'count: {plot_count}')
    return plot_x, plot_count, idx_list

def get_sorted_confidence_ppl_list(score_data, giveup_data, replace_data, origin, skip=False):
    confidence_ppl_list = []
    ppl_list = []
    new_score_data = []
    new_giveup_data = []
    for idx in range(len(score_data)):
        if skip and len(giveup_data[idx]['Res']) <= 1:
            continue
        try:
            confidence_ppl_list.append(get_confidence_ppl(giveup_data[idx], origin)) # 少数样本没有idx属性
            if len(giveup_data[idx]['Res']) <= 1:
                ppl_list.append(compute_ppl(replace_data[idx]))
            else:
                ppl_list.append(compute_ppl(giveup_data[idx]))
            new_score_data.append(score_data[idx])
            new_giveup_data.append(giveup_data[idx])
        except:
            print('Delete the sample without idx')
    sorted_ppl = sorted(enumerate(confidence_ppl_list), key=lambda x: x[1])
    return confidence_ppl_list, sorted_ppl, new_score_data, new_giveup_data, ppl_list


def sample_confidnece_ppl_giveup_plot(giveup_data, score_data, replace_data, prompt, origin=False):
    sample_dis = 200
    start = 0
    plot_x = []
    plot_confidence_ppl = []
    plot_ppl_certain = []
    plot_ppl_uncertain = []
    plot_giveup = []
    no_giveup = []
    no_giveup_em0 = []
    no_giveup_em1 = []
    giveup_em1 = []
    giveup_em0 = []
    giveup_ratio_em1 = []
    no_giveup_ratio_em0 = []
    uncertain_confidence_ppl = []
    certain_confidence_ppl = []
    max_confidence_ppl = []
    em0 = []
    em_list = []
    res_list = []
    confidence_ppl_list, sorted_ppl, score_data, giveup_data, ppl_list = get_sorted_confidence_ppl_list(score_data, giveup_data, replace_data, origin, skip=False)
    while start < len(score_data):
        if start + sample_dis < len(score_data):
            count = sample_dis
        else:
            count = len(score_data) - start
        
        if start >= 2400:
            for i in sorted_ppl[start: start + count]:
                res_list.append(giveup_data[i[0]])

        plot_x.append(str(start))
        temp_ppl_list_certain = [ppl_list[i[0]] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == False]
        temp_ppl_list_uncertain = [ppl_list[i[0]] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == True]
        temp_confidence_ppl_list = [confidence_ppl_list[i[0]] for i in sorted_ppl[start: start + count]]
        temp_giveup_list = [giveup_data[i[0]]['Giveup'] for i in sorted_ppl[start: start + count]]
        temp_no_giveup_em0 = [giveup_data[i[0]]['Giveup'] == False and score_data[i[0]]['EM'] == 0 for i in sorted_ppl[start: start + count]]
        temp_no_giveup_em1 = [giveup_data[i[0]]['Giveup'] == False and score_data[i[0]]['EM'] == 1 for i in sorted_ppl[start: start + count]]
        temp_no_giveup = [giveup_data[i[0]]['Giveup'] == False for i in sorted_ppl[start: start + count]]
        temp_giveup_em1 = [giveup_data[i[0]]['Giveup'] == True and score_data[i[0]]['EM'] == 1 for i in sorted_ppl[start: start + count]]
        temp_giveup_em0 = [giveup_data[i[0]]['Giveup'] == True and score_data[i[0]]['EM'] == 0 for i in sorted_ppl[start: start + count]]
        temp_certain_confidence_ppl = [confidence_ppl_list[i[0]] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == False]
        temp_uncertain_confidence_ppl = [confidence_ppl_list[i[0]] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == True]
        temp_em_list = [giveup_data[i[0]]['EM'] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == True]

        max_confidence_ppl.append(max([confidence_ppl_list[i[0]] for i in sorted_ppl[start: start + count]]))
        plot_giveup.append(sum(temp_giveup_list) / len(temp_giveup_list))
        plot_confidence_ppl.append(sum(temp_confidence_ppl_list) / len(temp_confidence_ppl_list))
        no_giveup_em0.append(sum(temp_no_giveup_em0) / len(temp_no_giveup_em0))
        no_giveup_em1.append(sum(temp_no_giveup_em1) / len(temp_no_giveup_em1))
        no_giveup.append(sum(temp_no_giveup) / len(temp_no_giveup))
        giveup_em1.append(sum(temp_giveup_em1) / len(temp_giveup_em1))
        giveup_em0.append(sum(temp_giveup_em0) / len(temp_giveup_em0))
        giveup_ratio_em1.append(giveup_em1[-1] / (giveup_em1[-1] + no_giveup_em1[-1]))
        no_giveup_ratio_em0.append(no_giveup_em0[-1] / (giveup_em0[-1] + no_giveup_em0[-1]))
        em0.append(no_giveup_em0[-1] + giveup_em0[-1])
        plot_ppl_certain.append(sum(temp_ppl_list_certain) / (len(temp_ppl_list_certain) + 0.0001))
        plot_ppl_uncertain.append(sum(temp_ppl_list_uncertain) / (len(temp_ppl_list_uncertain) + 0.0001))
        certain_confidence_ppl.append(sum(temp_certain_confidence_ppl) / len(temp_certain_confidence_ppl))
        if len(temp_uncertain_confidence_ppl) == 0:
            uncertain_confidence_ppl.append(0)
        else:
            uncertain_confidence_ppl.append(sum(temp_uncertain_confidence_ppl) / len(temp_uncertain_confidence_ppl))
        if len(temp_em_list) == 0:
            em_list.append(0)
        else:
            em_list.append(sum(temp_em_list) / len(temp_em_list))
        start += count
    write_jsonl(res_list, './res.jsonl')
    x_major_locator = MultipleLocator(2) # 在提供的plot_x上,每隔几个绘制一次横坐标刻度
    ax1 = plt.subplot(111)
    ax1.xaxis.set_major_locator(x_major_locator)

    plt.plot(plot_x, plot_confidence_ppl, color='#E9002D', label='mean_confidence_ppl')
    # plt.plot(plot_x, certain_confidence_ppl, color='#00B000', label='mean_certain_confidence_ppl')
    # plt.plot(plot_x, uncertain_confidence_ppl, color='#FFAA00', label='mean_uncertain_confidence_ppl')
    # plt.plot(plot_x, plot_ppl, color='#FFC61E', label='mean_ppl')
    plt.ylabel('confidence_ppl')
    plt.xlabel('count')
    plt.title(f"Sample={sample_dis}")
    plt.legend(loc=0)
    # ax2 = ax1.twinx()
    # ax2.xaxis.set_major_locator(x_major_locator)
    # plt.ylabel('score')

    # 绘制certain/uncertain比例
    # plt.plot(plot_x, no_giveup, color='purple', label='certain')
    # plt.plot(plot_x, no_giveup_em0, color='#00B000', label='no_giveup,EM=0')
    # plt.plot(plot_x, giveup_em1, color='#FFAA00', label='giveup,EM=1')

    # 绘制四象限比例
    # plt.plot(plot_x, no_giveup_em1, color='purple', label='certain_em1')
    # plt.plot(plot_x, no_giveup_em0, color='#00B000', label='certain_em0')
    # plt.plot(plot_x, giveup_em1, color='blue', label='uncertain_em1')
    # plt.plot(plot_x, giveup_em0, color='#FFAA00', label='uncertain_em0')

    # em=0/em=1的uncertain ratio
    # plt.plot(plot_x, em0, color='#FFAA00', label='em=0')
    # plt.plot(plot_x, giveup_ratio_em1, color='purple', label='uncertain in em=1')
    # plt.plot(plot_x, no_giveup_ratio_em0, color='#00B000', label='certain in em=0')

    #绘制em
    # plt.plot(plot_x, em_list, color='#FFAA00', label='em')

    # 绘制ppl
    plt.plot(plot_x, plot_ppl_certain, color='#FFAA00', label='mean_ppl for certain')
    plt.plot(plot_x, plot_ppl_uncertain, color='purple', label='mean_ppl for uncertain')

    plt.legend(loc=1)
    plt.show()

def dis_confidnece_ppl_giveup_plot(giveup_data, score_data, replace_data, prompt, origin=False):
    ppl_list, sorted_ppl, score_data, giveup_data = get_sorted_confidence_ppl_list(score_data, giveup_data, replace_data, origin)
    print(f'score data len: {len(score_data)}')
    ppl_range = range(0, int(max(ppl_list)) * 10, 20)
    ppl_dis = [float(i)/10 for i in ppl_range]
    plot_x = []
    plot_giveup = []
    print(ppl_dis)
    res_data = []
    # 得到分区的idx列表idx_list
    plot_x, plot_count, idx_list = get_idx_for_each_section(sorted_ppl, ppl_dis)
    for idx in range(len(idx_list)):
        item = idx_list[idx]
        temp_giveup = []
        for i in item:
            temp_giveup.append(giveup_data[i]['Giveup'])
        plot_giveup.append(sum(temp_giveup) / len(temp_giveup))

    x_major_locator = MultipleLocator(1)
    ax1 = plt.subplot(111)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.plot(plot_x, plot_count, color='#E9002D', label='count')
    plt.xlabel('confidence ppl')
    plt.ylabel('count')
    plt.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.xaxis.set_major_locator(x_major_locator)
    plt.plot(plot_x, plot_giveup, color='#00B000', label='giveup_ratio')
    plt.ylabel('ratio')
    plt.title('ppl_dis=2')
    plt.legend(loc=1)
    plt.show()

def get_digit_ratio(data, replace_data):
    res_list = []
    assert len(data) == len(replace_data)
    for idx in range(len(data)):
        ans = data[idx]['Res'] if len(data[idx]['Res']) > 1 else replace_data[idx]['Prediction']
        ans = remove_punc(ans).strip()
        res_list.append(is_digital(ans))
    print(f'digit ratio: {sum(res_list) / len(res_list)}')
    return res_list

def save_digit_data(data, sorted_ppl, digit_list):
    res_data = []
    for item in sorted_ppl:
        if digit_list[item[0]] == 1:
            res_data.append(data[item[0]])
    write_jsonl(res_data, './test_data/sorted_digit.jsonl')


def sample_dis_plot(giveup_data, score_data, replace_data, prompt, origin=False):
    sample_dis = 400
    start = 0
    out_path = f'./sample_dis_score_{prompt}_plot.png'
    plot_x = []
    plot_y_min = []
    plot_y_max = []
    plot_y_medien = []
    plot_giveup = []
    plot_em = []
    plot_f = []
    plot_has_answer = []
    plot_no_answer = []
    plot_len = []
    no_giveup_em0 = []
    no_giveup_em1 = []
    no_giveup = []
    giveup_em0 = []
    giveup_em1 = []
    giveup_ratio_em1 = []
    no_giveup_ratio_em0 = []
    em0 = []
    confidence_ppl_list = []
    confidence_ppl_certain = []
    confidence_ppl_uncertain = []
    digit_ratio = []
    res_data = []
    certain_em = []

    ppl_list, sorted_ppl, no_answer, score_data, giveup_data = get_sorted_ppl_list(score_data, replace_data, giveup_data, origin, False)
    spearman_em = []
    spearman_giveup = []
    spearman_ppl = []
    for item in sorted_ppl:
        spearman_em.append(giveup_data[item[0]]['has_answer'])
        spearman_giveup.append(giveup_data[item[0]]['Giveup'])
        spearman_ppl.append(ppl_list[item[0]])
    print(f'spearman ppl_has_answer: {get_spearman_coefficient(spearman_ppl, spearman_em)}')
    print(f'spearman ppl_giveup: {get_spearman_coefficient(spearman_ppl, spearman_giveup)}')
    # print(f'spearman has_answer_giveup: {get_spearman_coefficient(spearman_em, spearman_giveup)}')

    # digit_list = get_digit_ratio(giveup_data, replace_data)
    # save_digit_data(giveup_data, sorted_ppl, digit_list)
    for idx in range(len(ppl_list)):
        try:
            confidence_ppl_list.append(get_confidence_ppl(giveup_data[idx]))
        except:
            confidence_ppl_list.append(0)
    # print(confidence_ppl_list)
    # 统计各种数值
    while start < len(score_data):
        if start + sample_dis < len(score_data):
            count = sample_dis
        else:
            count = len(score_data) - start

        plot_x.append(str(start))
        temp_ppl_list = [ppl_list[i[0]] for i in sorted_ppl[start: start + count]]
        temp_giveup_list = [giveup_data[i[0]]['Giveup'] for i in sorted_ppl[start: start + count]]
        temp_em_list = [score_data[i[0]]['EM'] for i in sorted_ppl[start: start + count]]
        temp_f_list = [score_data[i[0]]['F1'] for i in sorted_ppl[start: start + count]]
        temp_has_answer_list = [score_data[i[0]]['has_answer'] for i in sorted_ppl[start: start + count]]
        temp_len_list = [get_answer_tokens(score_data[i[0]]) for i in sorted_ppl[start: start + count]]
        if len(no_answer) != 0: # 如果不是origin_prompt得到的结果
            temp_no_answer = [no_answer[i[0]] for i in sorted_ppl[start: start + count]]
        temp_no_giveup_em0 = [giveup_data[i[0]]['Giveup'] == False and score_data[i[0]]['EM'] == 0 for i in sorted_ppl[start: start + count]]
        temp_no_giveup_em1 = [giveup_data[i[0]]['Giveup'] == False and score_data[i[0]]['EM'] == 1 for i in sorted_ppl[start: start + count]]
        temp_no_giveup = [giveup_data[i[0]]['Giveup'] == False for i in sorted_ppl[start: start + count]]
        temp_giveup_em1 = [giveup_data[i[0]]['Giveup'] == True and score_data[i[0]]['EM'] == 1 for i in sorted_ppl[start: start + count]]
        temp_giveup_em0 = [giveup_data[i[0]]['Giveup'] == True and score_data[i[0]]['EM'] == 0 for i in sorted_ppl[start: start + count]]
        temp_confidence_ppl_certain = [confidence_ppl_list[i[0]] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == False]
        temp_confidence_ppl_uncertain = [confidence_ppl_list[i[0]] for i in sorted_ppl[start: start + count] if giveup_data[i[0]]['Giveup'] == True]
        # temp_digit_ratio = [digit_list[i[0]] for i in sorted_ppl[start: start + count]]

        plot_giveup.append(sum(temp_giveup_list) / len(temp_giveup_list))
        plot_em.append(sum(temp_em_list) / len(temp_em_list))
        plot_has_answer.append(sum(temp_has_answer_list) / len(temp_has_answer_list))
        plot_f.append(sum(temp_f_list) / len(temp_f_list))
        plot_y_min.append(min(temp_ppl_list))
        plot_y_max.append(max(temp_ppl_list) - min(temp_ppl_list))
        plot_y_medien.append(temp_ppl_list[int(count / 2)])
        plot_len.append(sum(temp_len_list) / len(temp_len_list))
        no_giveup_em0.append(sum(temp_no_giveup_em0) / len(temp_no_giveup_em0))
        no_giveup_em1.append(sum(temp_no_giveup_em1) / len(temp_no_giveup_em1))
        no_giveup.append(sum(temp_no_giveup) / len(temp_no_giveup))
        giveup_em1.append(sum(temp_giveup_em1) / len(temp_giveup_em1))
        giveup_em0.append(sum(temp_giveup_em0) / len(temp_giveup_em0))
        giveup_ratio_em1.append(giveup_em1[-1] / (giveup_em1[-1] + no_giveup_em1[-1] + 0.0001))
        no_giveup_ratio_em0.append(no_giveup_em0[-1] / (giveup_em0[-1] + no_giveup_em0[-1]))
        em0.append(no_giveup_em0[-1] + giveup_em0[-1])
        # digit_ratio.append(sum(temp_digit_ratio) / (len(temp_digit_ratio) + 0.0001))


        if len(no_answer) != 0:
            plot_no_answer.append(sum(temp_no_answer) / len(temp_no_answer))
        confidence_ppl_certain.append(sum(temp_confidence_ppl_certain) / (len(temp_confidence_ppl_certain)))
        confidence_ppl_uncertain.append(sum(temp_confidence_ppl_uncertain) / len(temp_confidence_ppl_uncertain))
        start += count
    # write_jsonl(res_data, './large_ppl.jsonl')
    certain_em = [no_giveup[i] - plot_em[i] for i in range(len(no_giveup))]
    print(certain_em)
    print(f'no_giveup: {no_giveup}')
    print(f'em score: {plot_em}')
    print(f'giveup_ratio: {plot_giveup}')
    print(f'no answer: {sum(no_answer)}')

    x_major_locator = MultipleLocator(1) # 在提供的plot_x上,每隔几个绘制一次横坐标刻度
    ax1 = plt.subplot(111)
    ax1.xaxis.set_major_locator(x_major_locator)

    plt.bar(plot_x, plot_y_max, bottom=plot_y_min, color='#FFAA00')
    plt.plot(plot_x, plot_y_medien, color='#E9002D', label='median_ppl')
    plt.ylabel('ppl')
    plt.xlabel('count')
    plt.title(f"Sample={sample_dis}")
    plt.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.xaxis.set_major_locator(x_major_locator)

    # # plot giveup rate
    # plt.plot(plot_x, plot_giveup, color='#00B000', label='uncertain_ratio')
    # plt.ylabel('ratio')
    # plt.legend(loc=1)
    # plt.show()

    # plot digit ratio
    # plt.plot(plot_x, certain_em, color='#00B000', label='confidence')

    #绘制四象限比例
    # plt.plot(plot_x, no_giveup_em1, color='purple', label='certain_em1')3
    # plt.plot(plot_x, no_giveup_em0, color='#00B000', label='certain_em0')
    # plt.plot(plot_x, giveup_em1, color='blue', label='uncertain_em1')
    # plt.plot(plot_x, giveup_em0, color='#FFAA00', label='uncertain_em0')

    # ppl
    # plt.plot(plot_x, confidence_ppl_certain, color='purple', label='certain_confidence_ppl')
    # plt.plot(plot_x, confidence_ppl_uncertain, color='#00B000', label='uncertain_confidence_ppl')
    # plt.legend(loc=1)
    # plt.show()
    # plt.savefig(out_path)    

    #plot em & f1 score
    plt.plot(plot_x, plot_has_answer, color='#00B000', label='has_answer')
    # plt.plot(plot_x, plot_f, color='#FFC61E', label='F1')
    plt.ylabel('score')
    plt.legend(loc=1)
    plt.show()
    # plt.savefig(out_path)   

def ppl_dis_plot(giveup_data, score_data, replace_data, prompt, origin=False, same_list=[]):
    out_path = f'./ppl_dis_score_{prompt}_plot.png'
    ppl_list = []
    giveup_plot = []
    plot_em = []
    plot_f = []
    plot_same = []
    plot_no_answer = []
    no_answer = []
    plot_len = []
    no_giveup_em0 = []
    no_giveup_em1 = []
    giveup_em0 = []
    giveup_em1 = []
    no_giveup = []
    res_idx = []
    ppl_list, sorted_ppl, no_answer, score_data, giveup_data = get_sorted_ppl_list(score_data, replace_data, giveup_data, origin, False)
    # 绘制范围(总的)
    ppl_range = range(0, int(max(ppl_list)) * 10, 10)
    ppl_dis = [float(i)/10 for i in ppl_range]

    #绘制范围(自由调节)
    # ppl_range = range(0, 210, 10)
    # ppl_dis = [float(i)/100 for i in ppl_range]
    print(ppl_dis)
    # 得到分区的idx列表idx_list
    plot_x, plot_count, idx_list = get_idx_for_each_section(sorted_ppl, ppl_dis)
    idx = 0
    # 根据分区的idx列表统计各种数值
    for item in idx_list:
        if plot_x[idx] == '1.0':
            res_idx = item
        idx += 1
        temp_giveup = []
        temp_em = []
        temp_f = []
        temp_same = []
        temp_no_answer = []
        temp_len = []
        temp_no_giveup_em0 = []
        temp_no_giveup_em1 = []
        temp_giveup_em0 = []
        temp_giveup_em1 = []
        temp_no_giveup = []
        for i in item:
            temp_giveup.append(giveup_data[i]['Giveup'])
            temp_em.append(score_data[i]['EM'])
            temp_f.append(score_data[i]['F1'])
            temp_len.append(get_answer_tokens(score_data[i]))
            temp_no_giveup_em0.append(giveup_data[i]['Giveup'] == False and score_data[i]['EM'] == 0)
            temp_no_giveup_em1.append(giveup_data[i]['Giveup'] == False and score_data[i]['EM'] == 1)
            temp_giveup_em0.append(giveup_data[i]['Giveup'] == True and score_data[i]['EM'] == 0)
            temp_giveup_em1.append(giveup_data[i]['Giveup'] == True and score_data[i]['EM'] == 1)
            temp_no_giveup.append(giveup_data[i]['Giveup'] == False)

            if len(no_answer) != 0:
                temp_no_answer.append(no_answer[i])
            if len(same_list) != 0:
                temp_same.append(same_list[i])
            
        giveup_plot.append(sum(temp_giveup) / len(temp_giveup))
        plot_em.append(sum(temp_em) / len(temp_em))
        plot_f.append(sum(temp_f) / len(temp_f))
        plot_len.append(sum(temp_len) / len(temp_len))
        no_giveup_em0.append(sum(temp_no_giveup_em0)/len(temp_no_giveup_em0))
        no_giveup_em1.append(sum(temp_no_giveup_em1)/len(temp_no_giveup_em1))
        giveup_em0.append(sum(temp_giveup_em0) / len(temp_giveup_em0))
        giveup_em1.append(sum(temp_giveup_em1) / len(temp_giveup_em1))
        no_giveup.append(sum(temp_no_giveup)/len(temp_no_giveup))
    # plot_same.append(sum(temp_same) / len(temp_same))
        if len(no_answer) != 0:
            plot_no_answer.append(sum(temp_no_answer) / len(temp_no_answer))
        if len(same_list) != 0:
            plot_same.append(sum(temp_same) / len(temp_same))
    write_jsonl(res_idx, './idx.jsonl')
    x_major_locator = MultipleLocator(1)
    # fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(111)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.plot(plot_x, plot_count, color='#E9002D', label='count')
    plt.legend(loc=2)
    plt.xlabel('ppl')
    plt.ylabel('count')
    plt.title('ppl_dis=1')
    
    
    ax2 = ax1.twinx()
    ax2.xaxis.set_major_locator(x_major_locator)
    # plt.plot(plot_x, giveup_plot, color='#00B000', label='uncertain_ratio')
    # if len(same_list) != 0:
    #     plt.plot(plot_x, plot_same, label='robust_ratio')
    # if len(no_answer) != 0:
    #     plt.plot(plot_x, plot_no_answer, color='#FFC61E', label='no_answer')
    # plt.savefig(out_path)
    # plt.plot(plot_x, no_giveup_em1, color='purple', label='certain_em1')
    # plt.plot(plot_x, no_giveup_em0, color='#00B000', label='certain_em0')
    # plt.plot(plot_x, giveup_em1, color='blue', label='uncertain_em1')
    # plt.plot(plot_x, giveup_em0, color='#FFAA00', label='uncertain_em0')
    plt.ylabel('ratio')
    plt.legend(loc=1)
    # plt.show()
    # plot em, f
    plt.plot(plot_x, plot_em, color='#00B000', label='EM')
    # plt.plot(plot_x, plot_f, color='#FFC61E', label='F1')
    plt.ylabel('score')
    plt.legend(loc=0)
    # plt.savefig(out_path)
    plt.show()