layer_mode='mid'
for model in llama2-chat-7b llama3-8b-instruct
do
    for mode in first last avg min ans dim_min dim_max
    do
        for seed in 0 42 100
        do
            python3 -u main.py \
                --model mlp \
                --data ./data/mmlu/$model/${layer_mode}_layer/zero-shot-chat/${layer_mode}_layer/$mode.pt \
                --label ./data/mmlu/$model/${layer_mode}_layer/zero-shot-chat/${layer_mode}_layer/labels.pt \
                --out_path ./data/mmlu/$model/${layer_mode}_layer/zero-shot-chat/${layer_mode}_layer/res/ \
                --seed $seed \
                --lr_rate 5e-5 \
                --epochs 30 \
                --which_gpu 1
        done
    done
done