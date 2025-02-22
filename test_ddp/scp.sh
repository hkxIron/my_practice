echo "HOME:$HOME"
# 同步数据
remote_project_path="/home/hkx/data/work/open/my_practice"
local_project_path="$HOME/work/open/project/"
user_ip="hkx@10.224.104.101"
rsync -av -e ssh --exclude='*.git' --exclude='.*' --exclude='*checkpoints*' --exclude='__pycache__/' --exclude='wandb/' ${user_ip}:${remote_project_path} $local_project_path
