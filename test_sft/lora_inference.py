import torch
  
import json
import math
import os, sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List
import pandas as pd
from peft import PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def inference(base_model_path:str, lora_path:str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
            trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16, # 加载半精度
            #device_map={"":0}, # 指定GPU 0
            device_map='cuda', # 指定GPU 0
            trust_remote_code=True, 
        )
    model.use_cache=True
    model.eval()
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    model.half()

    print(f"{model=}")
    print(f"{tokenizer=}")

    prompt = """请为以下关键词生成一条广告语。
类型#上衣*风格#简约*风格#性感*图案#线条*衣样式#针织衫*衣领型#一字领*衣门襟#系带"""

    input_ids = tokenizer(prompt, max_length=512, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=input_ids["input_ids"], max_new_tokens=256)
    print(tokenizer.decode(outputs[0]))
    print("inference end!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, help='')
    parser.add_argument('--lora_path', type=str, help='')
    # 添加一个参数来捕获剩余的所有参数
    parser.add_argument("unknown_args", nargs=argparse.REMAINDER, help="Unknown arguments")
    model_args = parser.parse_args()
    print(model_args)
    #print(unknown_args)
    inference(model_args.base_model_path, model_args.lora_path)