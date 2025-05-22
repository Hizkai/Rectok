python ../tokenizer/main.py \
    --data_name Baby \
    --normalize \
    --gpu_id 0 \
    --epochs 200 \
    --batch_size 1024 \
    --lr 1e-4 \
    --L 6 \
    --alpha 0.001 \
    --num_codes 1024 \
    --tau 0.25
    