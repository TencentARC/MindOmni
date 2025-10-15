export ASCEND_LAUNCH_BLOCKING=1
export HCCL_CONNECT_TIMEOUT=1800

EXP_NAME=exp_stage1

accelerate launch \
    --num_processes=1 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=20900 \
    train.py \
    --model_name_or_path /group/40043/yichengxiao/huggingface/model/OmniGen-v1 \
    --batch_size_per_device 16 \
    --condition_dropout_prob 0.01 \
    --lr 2e-5 \
    --use_dist \
    --freeze_omnigen \
    --dist_coef 1.0 \
    --diff_coef 1.0 \
    --x2i_length 2000000 \
    --json_file data/X2I_Data/main_instruction-mg.yaml \
    --image_path /group/40121/public_datasets/X2I_DATA \
    --model_llm /group/40043/yichengxiao/huggingface/model/Qwen2.5-VL-7B-Instruct \
    --t2i_ratio 1.0 \
    --t2i_json /group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset \
    --keys 'png' 'txt' \
    --num_workers 16 \
    --max_input_length_limit 12000 \
    --keep_raw_resolution \
    --max_image_size 256 \
    --val_img_size 512 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 100 \
    --epochs 1000 \
    --log_every 1 \
    --adam_weight_decay 0.0001 \
    --results_dir ./results/${EXP_NAME}

    # --use_dist \
    # --use_lora \
    # --lora_rank 16 \
    # --freeze_omnigen \

    # --sub_json /group/40034/yichengxiao/OmniGen/data/private_data/stage_2.yaml \
    # --sub_file_name /group/40121/public_datasets/X2I_DATA/X2I-subject-driven \

    # --t2i_json /group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset \
    # --file_name tar \
    # --keys 'png' 'txt' \

    # --use_longshort_t2i \
    # --t2i_json '/group/40033/seed_data/LAION-COCO-Recaption/*/part-*/**' \
    # --file_name tar \
    # --keys 'jpg' 'json' \

    # --sub_json /group/40034/yichengxiao/OmniGen/data/private_data/stage_2.yaml \
    # --sub_file_name /group/40121/public_datasets/X2I_DATA/X2I-subject-driven \


# rm /tmp/.unhold

    # --only_t2i \

# /group/40121/public_datasets/X2I_DATA/X2I-mm-instruction/pix2pix.jsonl
# /group/40121/public_datasets/X2I_DATA/X2I-mm-instruction/pix2pix
    # --pretrained_model results/exp5/checkpoints/0037500/qwen2phi.pt\

#
