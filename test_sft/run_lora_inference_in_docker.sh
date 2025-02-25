#!/usr/bin/env bash
. ../utils/shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/my_practice/"
model_path="${root_path}/open/hf_data_and_model/models/openbmb/MiniCPM3-4B/"
lora_model_path="${root_path}/trained_models/MiniCPM3-4B_mysft_lora/"

# docker image
img1="icr"
img2=".m"
img3="ice.cn"
img4='wsw/large-lm:1.0.15-2'
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image

max_gpu_num=2 # 最大限制多少个gpu
#test_max_gpu_num $max_gpu_num
if [ ! -d logs/ ]; then
    mkdir logs/
fi


port=$(python ../utils/get_free_port.py)
echo "port:$port"
#torchrun --rdzv-endpoint=localhost:${port} \
#export MASTER_PORT=${port} && \

#port=29501

device_list="4"
set -x
nohup docker run -i --rm --gpus '"device='${device_list}'"'  --name test_lora_inference --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/docker_workspace \
    -v ${model_path}:/docker_model_input_path \
    -v ${lora_model_path}:/docker_lora_model_path \
    -w /docker_workspace \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
python test_sft/lora_inference.py \
--base_model_path /docker_model_input_path \
--lora_path /docker_lora_model_path 
 " 2>&1 |tee logs/log_${time_str}.txt

set +x
echo "`date` 训练结束"

