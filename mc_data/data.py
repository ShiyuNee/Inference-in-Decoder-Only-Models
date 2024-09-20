import json
import random
from utils import has_answer, deal_judge
import csv
import re
random.seed(0)

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

def conver_hq_format(dir=''):
    data = json.loads(open('../hotpotqa/hotpot_train_v1.1.json').read())
    new_data = []
    for item in data:
        new_data.append({'question': item['question'], 'reference': item['answer'].split(', ')})
    print(len(new_data))
    write_jsonl(new_data, '../hotpotqa/hotpot_train.jsonl')

def write_2d_list_to_csv(filename, data):
    # 打开文件，使用 'w' 模式表示写入
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # 逐行写入数据
        for row in data:
            writer.writerow(row)

def clean_data(ref, gene_str):
    item_data = gene_str.replace('\n', '')
    # 使用正则表达式去除序号和多余空格
    cleaned_string = re.sub(r'\d+\.\s*', '', item_data).strip()
    # 以分号为分隔符进行分割并去除空白字符
    item_data = cleaned_string.split(':')
    # 生成格式可能是The answers are: , 也可能没有冒号
    item_data = item_data[0] if len(item_data) == 1 else ':'.join(item_data[1:])
    # 用完set顺序会变,需要控制以保证结果可复现
    # 选项不包含groun truth, 且不是拒绝回答
    gene_ans = sorted(list(set([item.strip() for item in item_data.split(';') if len(item) > 0 and not has_answer(ref, item) and not deal_judge(item)])))
    return gene_ans

def convert_qa_to_random_choices():
    dataset = 'hq'
    for mode in ['test', 'dev', 'train']:
        data = read_json(f'../{dataset}/{dataset}-{mode}.jsonl')
        csv_data = []
        for idx in range(len(data)):
            temp_answers = []
            idx_answer = {0:'A', 1:'B', 2:'C', 3:'D'}
            gt_choice='A'
            remain_idx = [item for item in range(len(data)) if item != idx]
            choose_answer = [data[item]['reference'][0] for item in random.sample(remain_idx, 3)]
            temp_answers.append(data[idx]['reference'][0])
            temp_answers += choose_answer
            random.shuffle(temp_answers)
            for ans_idx in range(len(temp_answers)):
                if temp_answers[ans_idx] == data[idx]['reference'][0]:
                    gt_choice = idx_answer[ans_idx]
            temp_answers.insert(0, data[idx]['question'])
            temp_answers.append(gt_choice)
            csv_data.append(temp_answers)
        write_2d_list_to_csv(f'../{dataset}/{dataset}-{mode}-random-choice.csv', csv_data)

def convert_qa_to_generated_choices():
    random.seed(0)
    mode = 'test'
    dataset = 'hq'
    data = read_json(f'../{dataset}/{dataset}-{mode}.jsonl')
    gene_data = read_json(f'../{dataset}/{dataset}_{mode}_llama8b_10_answers.jsonl')
    assert len(data) == len(gene_data)
    csv_data = []
    cnt = 0
    for idx in range(len(data)):
        temp_answers = []
        temp_answers.append(data[idx]['reference'][0])
        idx_answer = {0:'A', 1:'B', 2:'C', 3:'D'}
        gene_ans = clean_data(data[idx]['reference'], gene_data[idx]['Res'])
        sample_cnt = 3 # 出了ground truth, 还需要3个选项
        need_cnt = 0
        if len(gene_ans) >= 3:
            sample_data = random.sample(gene_ans, sample_cnt)
        else:
            cnt += 1
            sample_data = random.sample(gene_ans, len(gene_ans))
            need_cnt = 3 - len(gene_ans)
            remain_idx = [item for item in range(len(data)) if item != idx]
            # 生成的不够, 从其他问题答案中随机采样
            remain_sample = [data[item]['reference'][0] for item in random.sample(remain_idx, need_cnt)]
            sample_data += remain_sample
        sample_data.insert(0, data[idx]['reference'][0])
        random.shuffle(sample_data)
        for ans_idx in range(len(sample_data)):
            if sample_data[ans_idx] == data[idx]['reference'][0]:
                gt_choice = idx_answer[ans_idx]
        sample_data.insert(0, data[idx]['question'])
        sample_data.append(gt_choice)
        print(sample_data)
        csv_data.append(sample_data)
    print(cnt/len(gene_data))
    write_2d_list_to_csv(f'../{dataset}/{dataset}-{mode}-gene-choice_test.csv', csv_data)

def convert_qa_to_gene_none_data(path): 
    choice_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    data = []  
    res = []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        # 遍历并打印每一行
        for row in csv_reader:
            data.append(row)

    for idx in range(len(data)):
        gene_choice = [data[idx][1 + item] for item in range(4) if item != choice_idx[data[idx][-1]]] 
        new_data = [data[idx][0]] + gene_choice + ["None of above", 'D']
        res.append(new_data)
    out_path = path.replace('gene-choice', 'gene-none')
    write_2d_list_to_csv(out_path, res)

if __name__ == '__main__':
    path = '/Users/shiyuni/Documents/research/project/datasets/hq/hq-test-gene-choice_test.csv'
    convert_qa_to_gene_none_data(path)



