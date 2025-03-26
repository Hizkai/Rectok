# export CUDA_LAUNCH_BLOCKING=1
python3 ../tokenizer/main.py \
    --data_name Instruments23 \
    --gpu_id 0 \
    --lr 1e-4 \
    --loss_func div \
    --alpha 1.0 \
    --beta 0.001 \
    --hidden_dim 1024 \
    --activation topk \
    --output_label sae \
    --div_k 5 \
    --tied \
    --normalize \