import torch
import argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer)

def inference(input, model, tokenizer):
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(model.device)
    print(f"输入:\n    {input}")
    output_ids = model.generate(input_ids, num_beams=1, max_new_tokens=128)
    print(f"生成:\n    {tokenizer.decode(output_ids[0].tolist()[len(input_ids[0]):])}")

def test_infer(model_name:str):
    #model_name = "/home/hkx/data/work/hf_data_and_model/models/Qwen/Qwen2.5-3B-Instruct/"
    prompt = "一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评？"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #quant_conf = BitsAndBytesConfig()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 是否进行4bit量化, 需要gpu
        load_in_8bit=False,  # 是否进行8bit量化, 需要gpu
        bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
        bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
        bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
        bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
        llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
        llm_int8_has_fp16_weight=False,  # 是否启用混合精度
        llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
        llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
    )
    model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", 
                                                    quantization_config=quantization_config)
    #model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    model_native = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # 比较推理结果
    inference(prompt, model_8bit, tokenizer)
    inference(prompt, model_native, tokenizer)
    # 计算显存节约程度
    mem_fp16 = model_native.get_memory_footprint()
    mem_int8 = model_8bit.get_memory_footprint()
    print(mem_fp16/mem_int8)
    """
    输入:
    一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评？
    生成:
        这句话的立场看起来是赞扬的。它使用了诸如“传奇”、“不灭的神话”等积极词汇来描述这部电影，同时也将其视为一个重要和重要的文化标志，强调其在历史上的重要性。这些词汇和表述方式表明作者对这部电影持有正面评价，并认为它具有深远的影响和价值。因
    ，可以认为这句话的立场是赞扬的。不过，具体的解读可能还需要结合上下文和其他相关因素进行更深入分析。如果在某个特定语境下，这个句子被用作讽刺或反讽，则其立场可能是批评的。但从给出的信息来看，它倾向于表达
    输入:
        一个传奇的开端，一个不灭的神话，这不仅仅是一部电影，而是作为一个走进新时代的标签，永远彪炳史册。你认为这句话的立场是赞扬、中立还是批评？
    生成:
        这句话的立场倾向于赞扬。从描述中可以看出，这部电影被赋予了“传奇的开端”、“不灭的神话”的地位，这表明它在作者或叙述者眼中具有极高的价值和影响力。将电影比作新时代的标签和彪炳史册，则进一步强调了其重要性和历史意义。这些描述都是正面评价，
    示出作者对这部电影的高度赞赏之情。因此，这句话显然是带有赞扬态度的。如果要更准确地判断，需要更多的上下文信息，但根据给出的内容，它明显是积极正面的评价。 

    总结：这句话的立场是赞扬。<|endoftext|>
    5.310608466698831 # fp16相对于4bit占用内存是后者的5.3倍
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', default="~/data/work/hf_data_and_model/models/Qwen/Qwen2.5-3B/", type=str, help='')
    # 添加一个参数来捕获剩余的所有参数
    parser.add_argument("unknown_args", nargs=argparse.REMAINDER, help="Unknown arguments")

    model_args = parser.parse_args()
    print(model_args)
    #print(unknown_args)
    test_infer(model_args.base_model_path)