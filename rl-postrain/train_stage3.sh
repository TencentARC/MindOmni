source venus/init_program.sh
export HCCL_CONNECT_TIMEOUT=3600

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# on remote
data_paths="/group/40034/yichengxiao/VLM-R1/data/mindomni_stage3.jsonl"
image_folders="None"
model_path="/group/40043/yichengxiao/huggingface/model/MindOmni_qwen2.5vl"

is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-mindomni-lora_stage3" # TODO: change this to your own experiment name

TASK_TYPE="mindomni"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps

# attn_implementation='flash_attention_2'
attn_implementation='sdpa'
echo "attn_implementation: $attn_implementation"

qwen2phi_module_origin_path='/group/40034/yichengxiao/mindomni/results/exp_stage2/checkpoints/00060000/qwen2phi.pt'
echo $qwen2phi_module_origin_path

export WANDB_PROJECT="Reasoning_Generation_Repo"
export WANDB_API_KEY="your_wandb_key"

MACHINE_PARAMS="$1" # the string format of the machine params
echo "MACHINE_PARAMS: $MACHINE_PARAMS"
# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nproc_per_node=8 --master_port=29500 $MACHINE_PARAMS \
  src/open_r1/grpo_ust.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 100 \
    --bf16 \
    --attn_implementation $attn_implementation \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 50 \
    --num_generations 4 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.01 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --val_iter 50 \
    --diff_beta 0.06 \
    --qwen2phi_module_origin_path $qwen2phi_module_origin_path \

echo "Training completed for ${EXP_NAME}"

    # --reward_funcs accuracy format \
    # --beta 0.04 \
