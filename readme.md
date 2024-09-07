# 我对python的学习

# 阿里魔塔社区
https://modelscope.cn/my/overview

# hugging face (国内代理)
https://hf-mirror.com/

方法一：网页下载
在本站搜索，并在模型主页的Files and Version中下载文件。

方法二：huggingface-cli
huggingface-cli 是 Hugging Face 官方提供的命令行工具，自带完善的下载功能。

1. 安装依赖
pip install -U huggingface_hub
Copy
2. 设置环境变量
Linux
export HF_ENDPOINT=https://hf-mirror.com
Copy
Windows Powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
Copy
建议将上面这一行写入 ~/.bashrc。
3.1 下载模型
huggingface-cli download --resume-download gpt2 --local-dir gpt2
Copy
3.2 下载数据集
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
Copy
可以添加 --local-dir-use-symlinks False 参数禁用文件软链接，这样下载路径下所见即所得，详细解释请见上面提到的教程。


4. hfd 
需要安装 aria2c
hfd meta-llama/Llama-2-7b --hf_username YOUR_HF_USERNAME_NOT_EMAIL --hf_token YOUR_HF_TOKEN
hfd bigscience/bloom-560m

Download a model and exclude certain files (e.g., .safetensors):

hfd bigscience/bloom-560m --exclude *.safetensors
Download with aria2c and multiple threads:

hfd bigscience/bloom-560m




