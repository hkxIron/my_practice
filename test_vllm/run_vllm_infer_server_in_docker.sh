#!/usr/bin/env bash
. ../utils/shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/my_practice/"
model_path="${root_path}/open/hf_data_and_model/models/Qwen/Qwen2.5-3B/"

# docker image
img1="icr"
img2=".m"
img3="ice.cn"
#img4='wsw/large-lm:1.0.15-2'
img4='wsw/large-lm:1.0.15-4_vllm3' # 装了vllm的docker
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image

max_gpu_num=1 # 最大限制多少个gpu
test_max_gpu_num $max_gpu_num
if [ ! -d logs/ ]; then
    mkdir logs/
fi


port=$(python ../utils/get_free_port.py)
echo "port:$port"
#torchrun --rdzv-endpoint=localhost:${port} \
#export MASTER_PORT=${port} && \

#port=29501

#device_list="2"
set -x
docker run -i --rm --gpus '"device='${device_list}'"' -p 8000:8000 --name test_vllm_inference --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/docker_workspace \
    -v ${model_path}:/docker_model_input_path \
    -w /docker_workspace \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
. /opt/conda/etc/profile.d/conda.sh && \
conda activate vllm && \
vllm serve /docker_model_input_path " 

set +x
echo "`date` 预测结束"

<<EOF
# 启动之后可以在主机中进行curl测试

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/docker_model_input_path",
        "prompt": "9.11与9.9谁大呢",
        "max_tokens": 100,
        "temperature": 0
    }'

EOF