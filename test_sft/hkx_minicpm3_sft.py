# -*- coding: utf-8 -*-
import json
import os, sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List
import pandas as pd
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, Dataset, IterableDataset

import torch

print("sys path", sys.path)
print("cur work path:", os.getcwd())
#print("models path:", models_path)

#from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.trainer_utils import get_last_checkpoint

import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer,
                          default_data_collator)
from models.minicpm.modeling_minicpm import MiniCPM3ForCausalLM
from transformers.utils import PaddingStrategy

def show_paths():
    print("cur work path:", os.getcwd())
    print("sys path", sys.path)
    print(f"{os.environ=}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    qlora: bool = field(default=False)

"""
将多个input dict组成的list转换成一个dict,其中key不变，但value是多个原始数据组成的数组
eg:
input_dict_list:
[
    {
    "input_ids":[1,2,3,7,8,9],
    "label_ids":[-100,-100,-100,8,9,200],
    "attention_mask":[1,1,1],
    },
   {
    "input_ids":[1,2,3,7,8,9, 10, 11, 12],
    "label_ids":[-100,-100,-100,8,9,10,11,12, 200],
    "attention_mask":[1,1,1, 1,1,1],
    },
]

collator会先按max_seq_len进行padding后，再进行组成batch
BatchEncoding(
    data={
        "input_ids":[[1,2,3,7,8,9, 0,0,0],
                     [1,2,3,7,8,9, 10, 11, 12]],
        "label_ids":[[-100,-100,-100, 8,9,200], 
                     [-100, -100, -100, 8,9,10,11,12, 200]],
        "attention_mask":[[1,1,1, 1,1,1,0,0,0], 
                          [1,1,1, 1,1,1,1,1,1]],
    }
)
"""
# def get_my_collator(input_dict_list:List[Dict[str, list[int]]], max_seq_length:int, tokenizer: PreTrainedTokenizer)->BatchEncoding: # 注意：BatchEncoding实际上是个Dict[str, Tensor]
#     collator_data:BatchEncoding = tokenizer.pad(
#         input_dict_list,
#         padding_side='left',
#         padding= PaddingStrategy.MAX_LENGTH,
#         max_length=max_seq_length,
#         return_tensors="pt",
#     )
#     # collator:也可以完全自己计算，不需要使用tokenizer.pad函数
#     return collator_data

class SFTDataset:
    """Dataset for supervised fine-tuning.
    注意，并不是hf的Dataset
    """
    def __init__(
        self,
        data_path:str,
        tokenizer,
        max_seq_length=512,
        hint=False
    ):

        df = pd.read_json(data_path, lines=True)
        if hint:
            print(f"data columns:{list(df.columns)}")
            print(f"data len:{df.shape[0]}")
            print(f"head data:{df.head(3)=}")

        self.tokenizer = tokenizer
        self.model_max_length = max_seq_length
        self.ignore_index = -100
        # 构建hf dataset数据集
        self.dataset = datasets.Dataset.from_pandas(df)\
            .map(self.convert_tokens_to_ids, remove_columns=["prompt", "input", "output"])
        if hint:
            print(f"head data:{self.dataset[0:2]=}")
        self.dataset = self.dataset.map(remove_columns=["in_text", "out_text"])
        if hint:
            print(f"head data:{self.dataset[0:2]=}")
            print(f"dataset:{data_path} build done!")
    
    def convert_tokens_to_ids(self, example:Dict[str, str]):
        #input_ids = [self.tokenizer.bos_token_id]
        #label_ids = [self.ignore_index]
        
        prompt_text = example['prompt']
        input_text = example['input']
        output_text = example['output']

        # prompt部分不需要计算loss,因此使用ignore_index=-100 
        # 注意：
        # 1. tokenizer.__call__可以得到 input_ids+ attention_mask的Dict[str, List[int]]
        # 2.而tokenizer.encode只能得到 List[int], 所以此处只需要用encode
        prompt_input_ids = self.tokenizer.encode(prompt_text+input_text, add_special_tokens=False)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        input_ids = prompt_input_ids 
        label_ids = [self.ignore_index] * len(input_ids)
        
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        output_ids+=[eos_id]
        input_ids+=output_ids

        # 有个疑问是：如果在llm sft中，恰好input_text的最后一个token与output_text中第一个token可以组成一个新的token,
        # 那么len(tokenizer(input_text+output_text))<len(tokenizer(input_text))+len(tokenizer(output_text))了，就会产生id的错位，label_id也会错位，
        # 应该如何解决呢？
        # output需要计算loss, 使用原始的token_id
        label_ids+=output_ids

        #attention_mask = [1] * len(input_ids+label_ids)
        feature = {
            "in_text": prompt_text + input_text,
            "out_text": output_text,
            "input_ids": input_ids,
            "labels": label_ids,
            "attention_mask":[1]*len(input_ids)
        }
        return feature
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index:int):
        return self.dataset[index]

"""
将List[Dict]转为Dict[str, Tensor]
"""
def my_batch_padding_collator(examples: List[Dict[str, Any]], tokenizer, padding_side='left' , max_seq_len=1024) -> Dict[str, torch.Tensor]:
    """
        将List[Dict[str, Any]] 进行padding后转成Dict[str, Tensor]
        1. 将batch list中的多条样本变为一个batch中的单条样本
        2. 对一个batch中的样本进行padding
    """
    max_len_in_batch = min(max_seq_len, max([len(x["input_ids"]) for x in examples]))
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


def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = False,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""
    # minicpm3用的是llamaTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if use_lora and qlora:
        assert use_lora, "use_lora must be True when use_qlora is True"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 是否进行4bit量化, 需要gpu
            load_in_8bit=False,  # 是否进行8bit量化, 需要gpu
            bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
            bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
            bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
            bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
            llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
            llm_int8_has_fp16_weight=False,  # 是否启用混合精度
            #llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
            llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
        )
        model = MiniCPM3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            quantization_config=quantization_config, # 量化需要gpu支持才行
        )
    else:
        model = MiniCPM3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer

def show_some_data(dataset:Dataset, collate_fn: Callable, batch_size=2):
    # 打印点数据看下
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    sample = next(iter(dataloader))
    print(f"{sample=}")
    print("sample_batch input_ids shape:", sample['input_ids'].shape)
    print("sample_batch labels shape:", sample['labels'].shape)
    #print("sample_batch input_ids:", sample['input_ids'])
    #print("sample_batch labels:", sample['labels'])
    print("sample_batch attention_mask:", sample['attention_mask'])

    for idx in range(batch_size):
        input_ids = (sample["input_ids"][idx]).tolist()
        print(f"idx:{idx} {input_ids=}\n")
        print(f"idx:{idx} input_id to token map:{[(i, tokenizer.convert_ids_to_tokens(i)) for i in input_ids]}\n")

        labels = (sample["labels"][idx]).tolist()
        print(f"idx:{idx} {labels=}\n")
        print(f"idx:{idx} label id to token map:{[(i, tokenizer.convert_ids_to_tokens(i) if i>=0 else str(i)) for i in labels]}\n")

        print(f"idx:{idx} {sample['attention_mask'][idx].tolist()=}")
        print("="*50)


if __name__ == "__main__":
    show_paths()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16
    )

    max_seq_len = training_args.model_max_length
    train_dataset = SFTDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        hint=True
    )
    my_data_collator=lambda x: my_batch_padding_collator(x, tokenizer, max_seq_len=max_seq_len)


    eval_dataset = SFTDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        max_seq_length=training_args.model_max_length,
    )

    # debug
    show_some_data(train_dataset, my_data_collator)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=my_data_collator,
    )

    checkpoint = None
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    trainer.save_model()
    print("train done!")