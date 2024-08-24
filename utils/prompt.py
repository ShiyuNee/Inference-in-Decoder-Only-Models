from utils.utils import remove_punc
prompt_dict = {
    'qa': {
        'none': 'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'mc_qa': {
        'none': 'The following are multiple choice questions (with answers). Select the letter corresponding to the correct answer. Your response must be concise, without any irrelevant words. Do not include conversational words and do not provide any explanation.\n\n{question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_evidence': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_cot': {
        'none': 'Answer the following question based on your internal knowledge and let\'s think step by step.\n 1.Answer the question with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior': {
        'none': 'Are you sure to accurately answer the following question based on your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer \"Unknown\"\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nCan you answer the following question based on the given information or your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer \"Unknown\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'post': {
        'none': 'If you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}\nAnswer: {prediction}',
        'ra': 'Given the following information: \n{paras}\nIf you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is:',
    },
    'post_evidence': {
        'none': 'Give some evidence about why the answer may be right.\nQuestion: {question}{paras}\nAnswer: {prediction}',
        'ra': 'Given the following information: \n{paras}\nCan you judge the if the following answer about the question is correct based on the given information or your internal knowledge, if yes, you should answer True or False, if no, you should answer \"Unknown\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is: ',
    },
    'post_evidence_judge': {
        'none': 'If you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}\nAnswer: {prediction}\nEvidence:{revidence}',
        'ra': 'Given the following information: \n{paras}\nIf you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is: ',
    },
    'certain_prior': {
        'none': 'Answer the following question based on your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior_evidence': {
        'none': 'Answer the following question based on your internal knowledge with one or few words and explain why you give this answer. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Answer the following question based on the given information or your internal knowledge with one or few words and explain why you give this answer. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\". You will be punished if the answer is not right but you say \"certain\".\nGiven the following information: \n{paras}\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior_cot': {
        'none': 'Answer the following question based on your internal knowledge and let\'s think step by step.\n 1.Answer the question with one or few words.\n 2.If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nGenerate a short document that helps answer the following question based on the given information or your internal knowledge and answer the question with one or few words. If you are sure the answer is accurate and correct, please say \"certain\" after the answer. If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'paraphrase': {
        'none': 'Paraphrase the answer and you must keep the meaning.\nQuestion: {question}{paras}\nAnswer: {prediction}',
        'ra': 'Given the following information: \n{paras}\nIf you are sure the answer is accurate and correct, please say \"certain\". If you are not confident with the answer, please say \"uncertain\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is: ',
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
    prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction) + tail
    # 每个模型特有的prompt格式
    template_prompt = model_template_dict[args.model_name]
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt


