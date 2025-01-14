import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm

"""
使用与不使用kv cache可能导致结果不同

see: https://github.com/huggingface/transformers/issues/25420
"""

#base_path = "/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
#MODEL_PATH = f'{base_path}/models/MiniCPM-1B-sft-bf16'
base_path = "/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
MODEL_PATH = f'{base_path}/models/ahxt/LiteLlama-460M-1T'

#GEN_DEV = "cuda:0"
GEN_DEV = "cpu" # 即使使用cpu推理也是一样的

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(GEN_DEV)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True).to(GEN_DEV)


def get_input_ids(prompt: str) -> torch.Tensor:
    global model, tokenizer
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(GEN_DEV)
    return tokens


def tokens_to_text(tokens: torch.Tensor):
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)

PROMPT = "This is a "  # this is just a test prompt

gen_token_num = 20
# normal greedy search with HF Generate implementation
tokens = get_input_ids(PROMPT)
#lamaTokenizer.pad_token_id

# 使用hf generate的函数, 默认use_cache=True
tokens = model.generate(tokens, num_return_sequences=1, max_new_tokens=gen_token_num, use_cache=True)
generate_output_with_cache = tokens_to_text(tokens)[0]
print(f"hf generate with cache:{generate_output_with_cache=}")

tokens = get_input_ids(PROMPT)
tokens = model.generate(tokens, num_return_sequences=1, max_new_tokens=gen_token_num, use_cache=False)
generate_output_without_cache = tokens_to_text(tokens)[0]
print(f"hf generate without cache:{generate_output_without_cache=}")

# 不启用kv cache
# greedy decoding without caching
tokens = get_input_ids(PROMPT)
for _ in tqdm(range(gen_token_num)):
    with torch.no_grad():
        # 此处是手动取出logits, 未使用generate函数
        mout = model(tokens) # mout: [batch, seq_len, vocab_size], 默认不启用kv cache
    tokens = torch.hstack((tokens, torch.argmax(mout.logits[0, -1]).unsqueeze(0).unsqueeze(0)))
# 手动拼接，没有kv cache
without_cache = tokens_to_text(tokens)[0]
print(f"greedy decoding without cache: {without_cache=}") # 可以直接输出变量名

# 启用kv cache
# greedy decoding WITH caching
tokens = get_input_ids(PROMPT)
cached = None
for _ in tqdm(range(gen_token_num)):
    with torch.no_grad():
        if cached is None:
            mout = model(tokens, output_hidden_states=True, use_cache=True) # 启用kv cache
            cached = mout.past_key_values
        else:
            mout = model(tokens, past_key_values=cached, use_cache=True, output_hidden_states=True)
            cached = mout.past_key_values
    tokens = torch.hstack((tokens, torch.argmax(mout.logits[0, -1]).unsqueeze(0).unsqueeze(0)))

with_cache = tokens_to_text(tokens)[0]
print(f"greedy decoding with kv_cache:{with_cache=}")

# hf matches exactly
assert generate_output_without_cache == generate_output_with_cache

# this matches exactly
assert without_cache == generate_output_with_cache

# this does not!
assert without_cache == with_cache
