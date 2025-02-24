from typing import *

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer,
                          DataCollatorForLanguageModeling)
from transformers.utils import PaddingStrategy
from transformers import BatchEncoding, PreTrainedTokenizer

def test_batch_encode():
    text = [
        '你好，hello',
        '我是一名工程师',
        '我是一名工程师,已知，当前用户家里的所有房间信息和房间里的所有设备信息',
    ]

    model_path = '/home/hkx/data/work/hf_data_and_model/models/openbmb/MiniCPM-1B-sft-bf16'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"{tokenizer.pad_token=} {tokenizer.pad_token_id=}")
    # tokenizer.batch_encode_plus()
    batch_token_ids = tokenizer.__call__(text=text,
                                         add_special_tokens=False,
                                         padding=PaddingStrategy.MAX_LENGTH,
                                         truncation=True,
                                         max_length=10,
                                         return_tensors='pt')
    print(f"{batch_token_ids=}")
    """
    # 可以看出，默认是左padding
    tokenizer.pad_token='</s>' tokenizer.pad_token_id=2
    batch_token_ids={'input_ids': tensor([
    [    2,     2,     2,     2,     2,     2, 59320, 23523,    65, 17751],
    [    2,     2,     2,     2,     2,     2,     2, 14431, 20528, 14488],
    [14431, 20528, 14488, 59342,  6988,    65,  7387,  4194, 36440,  3216]]), 
    'attention_mask': tensor([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

    id to token map:[(2, '</s>'), (2, '</s>'), (2, '</s>'), (2, '</s>'), (2, '</s>'), (2, '</s>'), (59320, '▁'), (23523, '你好'), (65, '，'), (17751, 'hello')]

    batch_token_ids_plus={'input_ids': tensor([
        [    2,     2,     2,     2,     2,     2, 59320, 23523,    65, 17751],
        [    2,     2,     2,     2,     2,     2,     2, 14431, 20528, 14488],
        [14431, 20528, 14488, 59342,  6988,    65,  7387,  4194, 36440,  3216]]), 
        'attention_mask': tensor([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """

    sample_ids =batch_token_ids['input_ids'][0].tolist()
    print(f"id to token map:{[(i, tokenizer.convert_ids_to_tokens(i)) for i in sample_ids]}")


    """
    batch_encode_plus方法已废弃，请使用__call__代替

    结果与tokenizer.__call__相同
    """
    batch_token_ids_plus = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                  add_special_tokens=False,
                                                  padding_side='left', # 填充放在左边
                                                  padding=PaddingStrategy.MAX_LENGTH,
                                                  truncation=True,
                                                  max_length=10,
                                                  return_tensors='pt')
    print(f"{batch_token_ids_plus=}")

"""
将List[Dict]转为Dict[str, Tensor]
"""
def my_batch_padding_collator(examples: List[Dict[str, Any]], tokenizer, padding_side='left' , max_seq_len=1024) -> Dict[str, torch.Tensor]:
    """
        将List[Dict[str, Any]] 进行padding后转成Dict[str, Tensor]
        1. 将batch list中的多条样本变为一个batch中的单条样本
        2. 对一个batch中的样本进行padding
    """
    max_len_in_batch = min(max_seq_len+1, max([len(x["input_ids"]) for x in examples])) # 末尾多了一个eos
    padded_output = {}

    for example in examples:
        for key, value in example.items():
            if key == "labels":
                pad_id = -100
            elif key.startswith("attention"):
                pad_id = 0
            else:  # input token ids
                pad_id = tokenizer.pad_token_id
            # 截断
            value = value[:max_len_in_batch]
            # padding
            to_pad_ids = [pad_id]*(max_len_in_batch-len(value))
            if padding_side == "left":
                padded_value = to_pad_ids + value
            else:
                padded_value = value + to_pad_ids
            update_value = padded_output.setdefault(key, [])
            update_value.append(padded_value)
            padded_output[key] = update_value
    # 转为tensor_ids
    padded_tensor = {k:torch.LongTensor(v) for k,v in padded_output.items()} # 均为torch.int64
    return padded_tensor

def test_my_data_collator():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    my_data_collator = lambda x: my_batch_padding_collator(x, tokenizer)
    text_data = ['南京', '南京市', '南京市⻓江大桥']
    raw_tokens = [tokenizer(text, add_special_tokens=False) for text in text_data]
    print(f"{tokenizer.pad_token=} {tokenizer.pad_token_id=}")
    print(f"{raw_tokens=}")
    print(f"{tokenizer.convert_ids_to_tokens(1)}")

    data_batched = my_data_collator(raw_tokens)
    print(f"{data_batched=}")

    """
    tokenizer.pad_token='<unk>' tokenizer.pad_token_id=0
    # 3个句子，原来各自进行id->token,所以并未形成batch token,它们并未进行padding对齐
    raw_tokens=[{'input_ids': [29871, 30601, 30675], 
                 'attention_mask': [1, 1, 1]}, 
                {'input_ids': [29871, 30601, 30675, 30461], 
                'attention_mask': [1, 1, 1, 1]}, 
                {'input_ids': [29871, 30601, 30675, 30461, 229, 190, 150, 30775, 30257, 31749], 
                'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
                ]
    # 使用collator进行对齐后, 与上面手动batch的结果相同
    data_batched=
    {'input_ids': tensor([
    [    0,     0,     0,     0,     0,     0,     0,     29871, 30601, 30675],  # 注意：现在是left padding
    [    0,     0,     0,     0,     0,     0,     29871, 30601, 30675, 30461], 
    [    29871, 30601, 30675, 30461,   229,   190,   150, 30775, 30257, 31749]]), 
    'attention_mask': tensor([
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """

    # 下面对batch进行tokenizer,可以自动进行left padding后batch
    batch_token_ids = tokenizer.__call__(text=text_data,
                                         padding_side='left', # 填充放在左边
                                         add_special_tokens=False,
                                         padding=PaddingStrategy.MAX_LENGTH,
                                         truncation=True,
                                         max_length=10,
                                         return_tensors='pt')
    print(f"{batch_token_ids=}")
    """
    batch_token_ids={'input_ids': tensor([
        [    0,     0,     0,     0,     0,     0,     0, 29871, 30601, 30675],
        [    0,     0,     0,     0,     0,     0, 29871, 30601, 30675, 30461],
        [29871, 30601, 30675, 30461,   229,   190,   150, 30775, 30257, 31749]]),
    attention_mask': tensor([
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """



def test_data_collator():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    # DataCollatorForLanguageModeling
    # 这⾥的 tokenizer 选⽤的是 Qwen1.5 的，并⾮ LLaMA 的，只是做⼀个⽰意
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data = ['南京', '南京市', '南京市⻓江']
    raw_tokens = [tokenizer(text) for text in data]
    print(f'tokenizer.pad_token_id: {tokenizer.pad_token_id}\n')
    print("raw tokens:")
    print(raw_tokens)
    """
    raw_tokens:
    [{'input_ids': [1, 29871, 30601, 30675], 'attention_mask': [1, 1, 1, 1]}, 
     {'input_ids': [1, 29871, 30601, 30675, 30461], 'attention_mask': [1, 1, 1, 1, 1]}, 
     {'input_ids': [1, 29871, 30601, 30675, 30461, 229, 190, 150, 30775], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    ]


    after collator: 
    {'input_ids': 
        tensor([[    0,     0,     0,     0,     0,     1, 29871, 30601, 30675],
            [    0,     0,     0,     0,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]]), 
    'attention_mask': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    'labels': tensor([[ -100,  -100,  -100,  -100,  -100,     1, 29871, 30601, 30675],
            [ -100,  -100,  -100,  -100,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]])
    }
    """

    print("after collator:")
    print(data_collator(raw_tokens))


if __name__ == '__main__':
    #test_batch_encode()
    test_my_data_collator()
    #test_data_collator()