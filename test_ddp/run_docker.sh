#!/usr/bin/env bash
. ./shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/my_practice/test_ddp/"

# docker image
img1="icr"
img2=".m"
img3="ice.cn"
img4='wsw/large-lm:1.0.15-2'
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image

wandb_key="bdfc8b674cd322f967699975e89d431e82fcd317" # hkx wandb
max_gpu_num=2 # 最大限制多少个gpu
test_max_gpu_num $max_gpu_num
#device_list="3"
if [ ! -d logs/ ]; then
    mkdir logs/
fi

set -x
nohup docker run -i --rm --gpus '"device='${device_list}'"' --name train_simple_ddp --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/docker_workspace \
    -w /docker_workspace \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
export WANDB_DISABLED=false && \
export WANDB_PROJECT=simple_ddp && \
export WANDB_API_KEY=${wandb_key} && \
wandb login ${wandb_key} && \
python simple_ddp.py \
--enable_wandb True \
--is_debug False " 2>&1 |tee logs/log_${time_str}.txt

set +x
echo "`date` 训练结束"

