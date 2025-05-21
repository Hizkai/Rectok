export WANDB_MODE=disabled

DATASET=Games
BASE_MODEL=xxx
DATA_PATH=../dataset
OUTPUT_DIR=xxx
cd ..
torchrun --nproc_per_node=8 --master_port=33325 finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z3_bf16.json \
    --bf16 \
    --only_train_response \
    --index_file .index.json

cd convert
bash convert.sh $OUTPUT_DIR
cd ..

CKPT_PATH=$OUTPUT_DIR/final-checkpoint-12429
RESULTS_FILE=../results/$DATASET/results.json

torchrun --nproc_per_node=8 --master_port=23323 test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids 0,1,2 \
    --index_file .$desc.json