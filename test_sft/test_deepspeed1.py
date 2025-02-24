from typing import Dict
import nltk

import torch
import evaluate
import datasets
import numpy as np
import argparse
from functools import partial

from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.models.llama import LlamaForSequenceClassification
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

"""
    {
    "id": "13818513",
    "dialogue": "Amanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)",
    "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
    }
"""
def preprocess(examples:Dict[str, str], tokenizer, max_input_length:int, max_gen_length:int):
    dialogues = ["summarize:" + x for x in examples["dialogue"]]
    # summaries = [summ for summ in examples["summary"]]
    model_inputs = tokenizer(dialogues, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=max_gen_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

"""
该步骤负责对batch数据进行padding，这样每个batch都会有动态长度。
"""
def my_collate(features, tokenizer):
    batch_input_ids = [torch.LongTensor(feature["input_ids"]) for feature in features]
    batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) for feature in features]
    batch_labels = [torch.LongTensor(feature["labels"]) for feature in features]

    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

metric = evaluate.load("rouge")
def compute_metrics(eval_preds, tokenizer, sent_tokenize):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def train(data_path="", model_path=""):
    dataset_name = data_path+"/Samsung/samsum" # 数据集名称
    model_name=model_path + "/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA" # 模型名称

    max_input_length = 512
    max_gen_length = 128
    output_dir = "checkpoints"
    num_train_epochs = 5
    learning_rate = 5e-5
    deepspeed_config = "./ds_config.json" # deepspeed配置文件
    per_device_train_batch_size=1 # batch size设置为1，因为太大导致OOM
    per_device_eval_batch_size=1
    gradient_accumulation_steps=2 # 由于单卡的batch size为1，为了扩展batch size，使用梯度累加
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = datasets.load_dataset(dataset_name)
    print(f"dataset:{dataset}")
    """
    dataset:DatasetDict({
        train: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 14732
        })
        test: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 819
        })
        validation: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 818
        })
    })
    """
    print(dataset["train"][0])
    """
    {
    "id": "13818513",
    "dialogue": "Amanda: I baked  cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)",
    "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
    }
    """

    preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_input_length=max_input_length, max_gen_length=max_gen_length)
    tokenized_dataset = dataset.map(preprocess_fn, batched=True, remove_columns=["dialogue", "summary", "id"])

    input_ids_first = tokenized_dataset['train']['input_ids'][0]
    print(f"{tokenized_dataset=}") # 打印dataset
    print(f"{tokenized_dataset['train'][0:2]=}") # 打印dataset
    print(f"{input_ids_first=}") # 打印dataset
    print(f"id to token map:{[(i, tokenizer.convert_ids_to_tokens(i)) for i in input_ids_first]}")
    print(f"origin text:{dataset['train'][0]}")

    # 用于测试的代码, 
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    test_dataloader = DataLoader(tokenized_dataset["test"], shuffle=False, batch_size=2, collate_fn=collate_fn)
    test_batch = next(iter(test_dataloader)) # 一定要加iter
    print(f"{test_batch=}") # 打印dataset

    # 加载模型
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # 用于测试的代码
    dataloader = DataLoader(tokenized_dataset["test"], shuffle=False, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    output_before_train = model(**batch)
    print(f"{output_before_train=}") # 打印dataset

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=1, # 防止评估时导致OOM
        predict_with_generate=True,
        fp16=False,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        # logging & evaluation strategies
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=50, # 每50个step打印一次log
        evaluation_strategy="steps",
        eval_steps=500, # 每500个step进行一次评估
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        deepspeed=deepspeed_config, # deepspeed配置文件的位置
        report_to="wandb"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # 打印验证集上的结果
    print(trainer.evaluate(tokenized_dataset["validation"]))
    # 打印测试集上的结果
    print(trainer.evaluate(tokenized_dataset["test"]))
    # 保存最优模型
    trainer.save_model("best")
    print("train训练结束")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/hkx/data/work/hf_data_and_model/datas/', help='')
    parser.add_argument('--model_path', type=str, default='/home/hkx/data/work/hf_data_and_model/models/', help='')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    print(f"args:{args}")

    # 现在 torchrun 负责在各个 GPU 上生成进程并执行，不再需要 mp.spawn 了
    train(args.data_path, args.model_path)