python ../tokenizer/main.py \
    --data_name Games \
    --normalize \
    --gpu_id 0 \
    --epochs 100 \
    --patience 200 \
    --batch_size 1024 \
    --lr 1e-4 \
    --L 5 \
    --alpha 0.001 \
    --num_codes 1024 \
    --tau 0.25
    