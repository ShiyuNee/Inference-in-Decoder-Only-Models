from utils.utils import remove_punc
prompt_dict = {
    'qa': {
        'none': 'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_cot': {
        'none': 'Answer the question by briefly explaining your reasoning with one or few sentences, then provide the final answer.\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '\nAnswer: ',
    },
    'qa_more': {
        'none': 'Generate 10 possible answers for the following question, each separated by a semicolon. These 10 answers must be different, and your response should be as concise as possible, with no irrelevant words beyond the answers.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'mc_qa': {
        'none': 'The following are multiple choice questions (with answers){subject}. Select the correct answer without any irrelevant words. Do not include conversational words and do not provide any explanation.\n\n{question}{paras}{prediction}',
        # 'none': 'The following are multiple choice questions (with answers) {subject}.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'mc_qa_cot': {
        'none': 'The following are multiple choice questions (with answers){subject}. Briefly explain your reasoning with one or few sentences and choose the correct answer.\n\n{question}{paras}{prediction}',
        # 'none': 'The following are multiple choice questions (with answers) {subject}.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'mc_qa_evidence': {
        'none': 'The following are multiple choice questions (with answers){subject}. Select the correct answer without any irrelevant words and explain why you choose this answer briefly.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_evidence': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer briefly.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
}

model_template_dict = {
    'llama2-7b-chat':{
        'prefix': '<s>[INST] <<SYS>>\nYou are a helpful assistant<</SYS>>',
        'end': '[/INST]'
    },
    'llama3-8b-instruct':{
        'prefix': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for answering factual questions<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n',
        'end': "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    }
}

def get_prompt(sample, args):
    paras = ""
    ref_key = 'question'
    prompt = prompt_dict[args.type]['none'] # prior
    if args.ra != 'none':
        ra_dict = args.ra
        i = 0
        doc = []
        for k, v in ra_dict.items():
            v = min(v, len(sample[k]))
            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                i += 1
        paras = '\n'.join(doc)
        prompt = prompt_dict[args.type]['ra']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prediction = sample['Res'] if 'post' in args.type else ""
    # mmlu基础模板里带subject
    # print(args.task)
    if args.task == 'mmlu' or args.task == 'tq':
        prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction, subject=args.subject) + tail
    else:
        prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction) + tail
    # 每个模型特有的prompt格式
    template_prompt = model_template_dict[args.model_name]
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt


