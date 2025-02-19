from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import config
import torch

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def flush():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def test1():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')
    # device_map='auto', 流水线并行
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
    print(result)
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

def test_8bit():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')
    # device_map='auto', 流水线并行
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True , device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
    print(result)
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

def test_4bit():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')
    # device_map='auto', 流水线并行
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True , device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
    print(result)
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

def test_no_flashatten():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    import time
    long_prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"*5
    start_time = time.time()
    result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]
    print(f"Generated in {time.time() - start_time} seconds.")

def test_with_flashatten():
    #print(transformers.__version__)
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    import time
    long_prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"*5
    start_time = time.time()
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]
    print(f"Generated in {time.time() - start_time} seconds.")

def test_generate_no_kv_cache():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device) # input_ids: [batch, seq_len]

    for _ in range(5):
        next_logits = model(input_ids)["logits"][:, -1:]
        next_token_id = torch.argmax(next_logits, dim=-1)
        # 没有kv cached的解码，需要手动将next_token_id与input_ids拼接起来
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        print("shape of input_ids:", input_ids.shape)

    generated_text = tokenizer.batch_decode(input_ids[:, -5:])
    print("generated_text:", generated_text)

def test_generate_with_kv_cache():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps

    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    # 需要传入kv cache,并返回kv cache
    generated_tokens = []
    next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    past_key_values = None  # past_key_values is the key-value cache, kv cache初始化为None,后面动态增加

    for _ in range(5):
        # 每次只输入下一个token
        next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
        # next_logits:[batch, seq_len, vocab_size]
        # -> [batch, last_token=1, vocab_size]
        next_logits = next_logits[:, -1:]
        next_token_id = torch.argmax(next_logits, dim=-1)

        print("shape of input_ids", next_token_id.shape)
        # 取出第0层的key的cache来看看
        print("layer num of key-value cache", len(past_key_values))  # 第1个0为第0层，第2个0为第key, past_key_values are of shape [num_layers, 0 for k and 1 for v, batch_size, length, hidden_dim]
        print("length of key cache shape of layer_0", past_key_values[0][0].shape)  # 第1个0为第0层，第2个0为第key, past_key_values are of shape [num_layers, 0 for k and 1 for v, batch_size, length, hidden_dim]
        print("length of value cache shape of layer_0", past_key_values[0][1].shape)  # 第1个0为第0层，第2个0为第key, past_key_values are of shape [num_layers, 0 for k and 1 for v, batch_size, length, hidden_dim]
        generated_tokens.append(next_token_id.item())

        """
        输出：
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 20, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 20, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 21, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 21, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 22, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 22, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 23, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 23, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 24, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 24, 64])
        generated_text: ['\n', 'def', ' transform', '_', 'bytes']
        
        """

    generated_text = tokenizer.batch_decode(generated_tokens)
    print("generated_text:", generated_text)

def test_multiturn_kv_cache():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps

    # Generation as usual
    system_prompt = "your are a helpful assistant."
    prompt = system_prompt + "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer: Here"
    model_inputs = tokenizer(prompt, return_tensors='pt')
    generation_output = model.generate(**model_inputs, max_new_tokens=60, return_dict_in_generate=True)
    decoded_output = tokenizer.batch_decode(generation_output.sequences)[0]
    print(f"{generation_output=}")

    # 第二轮对话时，使用上一轮的kv cache
    # Piping the returned `past_key_values` to speed up the next conversation round
    prompt = decoded_output + "\nQuestion: How can I modify the function above to return Mega bytes instead?\n\nAnswer: Here"
    model_inputs = tokenizer(prompt, return_tensors='pt')
    generation_output = model.generate(
        **model_inputs,
        past_key_values=generation_output.past_key_values, # 直接使用上一轮的kv cache
        max_new_tokens=60,
        return_dict_in_generate=True
    )
    print(tokenizer.batch_decode(generation_output.sequences)[0][len(prompt):])

if __name__ == '__main__':
    #test1()
    #test_8bit()
    #test_4bit()
    #test_no_flashatten()
    #test_with_flashatten()
    #test_generate_no_kv_cache()
    test_generate_with_kv_cache()
    test_multiturn_kv_cache()