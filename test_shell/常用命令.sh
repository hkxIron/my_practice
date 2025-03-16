在当前目录下递归查找所有json文件中含有"Lamb"关键字的行 
 grep -i "Lamb" $(find . -name '*.json')

查看torch版本
python -c 'import torch; print(f"torch: {torch.__version__}")'
python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'