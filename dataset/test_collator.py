from typing import *

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer)
from transformers.utils import PaddingStrategy
from transformers import BatchEncoding, PreTrainedTokenizer


if __name__ == '__main__':
    text = [
        '你好，hello',
        '我是一名工程师',
        '我是一名工程师,已知，当前用户家里的所有房间信息和房间里的所有设备信息',
    ]

    model_path='/home/hkx/data/work/hf_data_and_model/models/MiniCPM-1B-sft-bf16'
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.batch_encode_plus()
    batch_token_ids = tokenizer.__call__(text=text,
                                         add_special_tokens=False,
                                         padding=PaddingStrategy.MAX_LENGTH,
                                         truncation=True,
                                         max_length=10,
                                         return_tensors='pt')
    print("batch_token_ids:")
    print(batch_token_ids)
    """
    {'input_ids': tensor([[    2,     2,     2,     2,     2,     2, 59320, 23523,    65, 17751],
        [    2,     2,     2,     2,     2,     2,     2, 14431, 20528, 14488],
        [14431, 20528, 14488, 59342,  6988,    65,  7387,  4194, 36440,  3216]]), 
        
     'attention_mask': tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """

    batch_token_ids = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                         add_special_tokens=False,
                                         padding=PaddingStrategy.MAX_LENGTH,
                                         truncation=True,
                                         max_length=10,
                                         return_tensors='pt')
    print("batch_token_ids:")
    print(batch_token_ids)
