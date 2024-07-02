task=mmlu
source=../datasets/mmlu/
outfile=./res/mmlu/zero-shot-evidence/

# task=tq
# source=./truthfulqa
# outfile=./res/tq/zero-shot/


type=qa_evidence

python run_mmlu.py \
    --source $source \
    --type $type \
    --ra none \
    --outfile $outfile \
    --n_shot 0 \
    --model_path ../models/llama2-7B-chat \
    --batch_size 4 \
    --task $task \
    --max_new_tokens 16