import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import config

def inference(payload, model, tokenizer):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(model.device)
    print(f"输入:\n    {payload}")
    model.eval()
    logits = model.generate(input_ids, num_beams=1, max_new_tokens=128)
    res = tokenizer.decode(logits[0].tolist()[len(input_ids[0]):])
    print(f"生成:\n{res}")
    return res

#model_name = "bigscience/bloomz-7b1-mt"
base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
payload = "一个传奇的开端，作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评？"
print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')

"""
Naive pipeline parallelism is supported out of the box. 
For this, simply load the model with device="auto" which will automatically 
place the different layers on the available GPUs as explained here. 
Note, however that while very effective, this naive pipeline parallelism does 
not tackle the issues of GPU idling. 
For this more advanced pipeline parallelism is required as explained here.
"""
# device='auto', 就是流水线并行，将不同层放在不同的gpu上
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if torch.cuda.is_available():
    model_8bit = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", load_in_8bit=True) # device_map="auto",简单的流水线并行
else:
    model_8bit = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16) # device_map="auto",简单的流水线并行
model_8bit.eval()

model_native = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float32)
#model_native = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto") # 如果不指定类型，默认为float32
model_native.eval()

# 计算显存节约程度
mem_fp16 = model_native.get_memory_footprint()
mem_int8 = model_8bit.get_memory_footprint()
print(mem_fp16/mem_int8)

# 比较推理结果
res1 = inference(payload, model_8bit, tokenizer)
res2 = inference(payload, model_native, tokenizer)
assert res1 == res2