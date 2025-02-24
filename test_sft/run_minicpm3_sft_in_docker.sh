#!/usr/bin/env bash
. ../utils/shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/my_practice/"
model_path="${root_path}/open/hf_data_and_model/models/openbmb/MiniCPM3-4B/"
model_out_path="${root_path}/open/hf_data_and_model/models/openbmb/MiniCPM3-4B_mysft/"

# docker image
img1="icr"
img2=".m"
img3="ice.cn"
img4='wsw/large-lm:1.0.15-2'
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image

max_gpu_num=2 # 最大限制多少个gpu
test_max_gpu_num $max_gpu_num
if [ ! -d logs/ ]; then
    mkdir logs/
fi


wandb_key="bdfc8b674cd322f967699975e89d431e82fcd317" # hkx wandb
port=$(python ../utils/get_free_port.py)
echo "port:$port"
#torchrun --rdzv-endpoint=localhost:${port} \
#export MASTER_PORT=${port} && \

#port=29501

set -x
nohup docker run -i --rm --gpus '"device='${device_list}'"'  --name test_minicpm3_sft --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/docker_workspace \
    -v ${model_path}:/docker_model_input_path \
    -v ${model_out_path}:/docker_model_out_path \
    -w /docker_workspace \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
export WANDB_DISABLED=false && \
export WANDB_PROJECT=simple_ddp && \
export WANDB_API_KEY=${wandb_key} && \
wandb login ${wandb_key} && \
deepspeed \
--master_port=${port} \
test_sft/hkx_minicpm3_sft.py \
--deepspeed test_sft/deepspeed_bf16_zero2.json \
--model_name_or_path /docker_model_input_path \
--report_to wandb \
--output_dir /docker_model_out_path \
--train_data_path test_sft/data/AdvertiseGenChatML/train.jsonl \
--eval_data_path test_sft/data/AdvertiseGenChatML/dev.jsonl \
--learning_rate 5e-5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 2 \
--bf16 \
--gradient_accumulation_steps 2 \
--warmup_steps 100 \
--max_steps 2000 \
--weight_decay 0.01 \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy steps \
--save_steps 500 --seed 42 \
--log_level info \
--logging_strategy steps \
--logging_steps 10
 " 2>&1 |tee logs/log_${time_str}.txt

set +x
echo "`date` 训练结束"

