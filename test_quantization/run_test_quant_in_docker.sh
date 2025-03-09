#!/usr/bin/env bash
. ../utils/shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/my_practice/"
#model_path="${root_path}/open/hf_data_and_model/models/Qwen/Qwen2.5-7B-Instruct/"
model_path="${root_path}/open/hf_data_and_model/models/Qwen/Qwen2.5-3B/"

# docker image
img1="icr"
img2=".m"
img3="ice.cn"
img4='wsw/large-lm:1.0.15-2'
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image


if [ ! -d logs/ ]; then
    mkdir logs/
fi


port=$(python ../utils/get_free_port.py)
echo "port:$port"
#torchrun --rdzv-endpoint=localhost:${port} \
#export MASTER_PORT=${port} && \

#port=29501

#device_list="4"
auto_find_gpu=1
if [ $auto_find_gpu -eq 1 ];then
    max_gpu_num=1 # 最大限制多少个gpu
    test_max_gpu_num $max_gpu_num
else
    device_list="2"
fi

set -x
nohup docker run -i --rm --gpus '"device='${device_list}'"'  --name test_quant --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/docker_workspace \
    -v ${model_path}:/docker_model_input_path \
    -w /docker_workspace \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
python test_quantization/test_hf_quantization.py \
--base_model_path /docker_model_input_path \
 " 2>&1 |tee logs/log_${time_str}.txt

set +x
echo "`date` 预测结束"

