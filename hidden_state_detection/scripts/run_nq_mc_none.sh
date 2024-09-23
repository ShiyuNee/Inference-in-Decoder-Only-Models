layer_mode='mid'
for model in llama2-chat-7b llama3-8b-instruct qwen2
do
    for chat_mode in zero-shot-none
    do
        for mode in first last avg ans
        do
            for seed in 0 42 100
            do
            python3 -u main.py \
                --model mlp \
                --data ./data/nq-mc/$model/${layer_mode}_layer/${chat_mode}/${layer_mode}_layer/${mode}_train.pt \
                --label ./data/nq-mc/$model/${layer_mode}_layer/${chat_mode}/${layer_mode}_layer/train_labels.pt \
                --out_path ./data/nq-mc/$model/${layer_mode}_layer/${chat_mode}/${layer_mode}_layer/res/ \
                --seed $seed \
                --lr_rate 5e-5 \
                --epochs 30 \
                --which_gpu 0
            done
        done
    done
done

