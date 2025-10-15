export ASCEND_LAUNCH_BLOCKING=1
export HCCL_CONNECT_TIMEOUT=1800

EXP_NAME=exp_stage2

accelerate launch \
    --num_processes=1 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=20900 \
    train.py \
    --model_name_or_path /group/40043/yichengxiao/huggingface/model/OmniGen-v1 \
    --batch_size_per_device 1 \
    --condition_dropout_prob 0.01 \
    --lr 5e-5 \
    --use_dist \
    --dist_coef 0.0 \
    --diff_coef 1.0 \
    --use_lora \
    --lora_rank 16 \
    --model_llm /group/40043/yichengxiao/huggingface/model/Qwen2.5-VL-7B-Instruct \
    --t2i_ratio 1.0 \
    --t2i_json /group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset \
    --keys 'png' 'txt' \
    --sub_json /group/40034/yichengxiao/MindOmni/data/stage_2.yaml \
    --num_workers 16 \
    --max_input_length_limit 12000 \
    --keep_raw_resolution \
    --max_image_size 512 \
    --val_img_size 512 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 100 \
    --epochs 1000 \
    --log_every 1 \
    --adam_weight_decay 0.0001 \
    --pretrained_model results/exp_stage1/checkpoints/00100000/qwen2phi.pt \
    --results_dir ./results/${EXP_NAME}
