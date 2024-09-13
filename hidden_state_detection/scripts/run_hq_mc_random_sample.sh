layer_mode='mid'
for model in llama3-8b-instruct
do
    for chat_mode in zero-shot-random zero-shot-gene
    do
        for mode in first last avg
        do
            for seed in 0 42 100
            do
            python3 -u main.py \
                --model mlp \
                --data ./data/hq-mc/$model/${layer_mode}_layer/${chat_mode}/${layer_mode}_layer/sample_${mode}_train.pt \
                --label ./data/hq-mc/$model/${layer_mode}_layer/${chat_mode}/${layer_mode}_layer/sample_train_labels.pt \
                --out_path ./data/hq-mc/$model/${layer_mode}_layer/${chat_mode}/${layer_mode}_layer/sample_res/ \
                --seed $seed \
                --lr_rate 5e-5 \
                --epochs 50 \
                --which_gpu 1
            done
        done
    done
done

