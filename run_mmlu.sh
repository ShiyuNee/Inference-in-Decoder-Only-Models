# python run_mmlu.py --source ../datasets/mmlu/ --type qa --ra none --outfile ./res/mmlu/zero-shot-hidden/ --n_shot 0 --model_path ../models/llama2-7B-chat --batch_size 2 --task mmlu --max_new_tokens 1

deepspeed --num_gpus 1 run_mmlu.py --source ../datasets/mmlu/ --type qa --ra none --outfile ./res/mmlu/zero-shot-hidden/ --n_shot 0 --model_path ../models/llama2-7B-chat --batch_size 2 --task mmlu --max_new_tokens 1