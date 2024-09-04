# pytorch-quantization-demo

code source:
https://github.com/Jermmy/pytorch-quantization-demo 

A simple network quantization demo using pytorch from scratch. This is the code for my [tutorial](https://mp.weixin.qq.com/s?__biz=Mzg4ODA3MDkyMA==&mid=2247483692&idx=1&sn=3e28db4881d591f4e6a66c83d4213823&chksm=cf81f74bf8f67e5d0f2a98fd7bf7a91864d14010d88a5ed89120b7b4fcd94fc34789f0d0db9a&token=680347690&lang=zh_CN#rd) about network quantization written in Chinese. 

也欢迎感兴趣的读者关注我的知乎专栏：[大白话模型量化](https://zhuanlan.zhihu.com/c_1258047709686231040)

# 手动实现后训练量化
## 1. 训练全精度网络
python tran.py

## 2. 后训练量化
python  post_training_quantize.py


# 量化感知训练
python quantization_aware_training.py


