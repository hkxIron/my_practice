# SPDX-License-Identifier: Apache-2.0
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import vllm
from vllm import LLM, SamplingParams
import argparse
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer)


def serve(base_model_path:str):

    # Create a sampling params object.
    #sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 4bit量化
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # 是否进行4bit量化, 需要gpu
    #     load_in_8bit=False,  # 是否进行8bit量化, 需要gpu
    #     bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
    #     bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
    #     bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
    #     bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
    #     llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
    #     llm_int8_has_fp16_weight=False,  # 是否启用混合精度
    #     llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
    #     llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
    # )

    # # 使用 bitsandbytes 以 8 位量化加载模型
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     quantization_config = quantization_config,
    #     #load_in_8bit=True,  # 启用 8 位量化
    #     device_map="auto",  # 自动分配设备
    #     torch_dtype=torch.float16,  # 仍然使用 FP16 进行计算
    # )

    print(f"model path:{base_model_path}")

    # 加载 tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Create an LLM.
    llm = LLM(model=base_model_path, 
              tokenizer=tokenizer,
                quantization="gptq",
                quantization_config={
                    "algorithm": "nf4",  # 选择量化算法
                    "group_size": 128,   # 分组大小
                }
    )
    print(f"llm model:{llm}")

    # 启动服务
    llm.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', default="~/data/work/hf_data_and_model/models/Qwen/Qwen2.5-3B/", type=str, help='')
    # 添加一个参数来捕获剩余的所有参数
    parser.add_argument("unknown_args", nargs=argparse.REMAINDER, help="Unknown arguments")

    model_args = parser.parse_args()
    print(model_args)
    #print(unknown_args)
    serve(model_args.base_model_path)