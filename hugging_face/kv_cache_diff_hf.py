import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm

"""
使用与不使用kv cache可能导致结果不同

see: https://github.com/huggingface/transformers/issues/25420
"""

base_path = "/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
MODEL_PATH = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
ds = load_dataset(path="/home/hkx/data/work/hf_data_and_model/datas/TinyStoriesV2/", split="validation", streaming=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

TOTAL_NUM_SAMPLES = 10
INPUT_LEN = 64

# model = AutoModelForCausalLM.from_pretrained(model_name)
ds_iterator = iter(ds.take(TOTAL_NUM_SAMPLES))
max_diffs = {}

for _ in tqdm(range(TOTAL_NUM_SAMPLES)):
    next_data = next(ds_iterator)["text"]
    # 每次只有一条样本
    all_input_ids = tokenizer([next_data], return_tensors="pt", max_length=INPUT_LEN, truncation=True).input_ids.to(model.device)

    # process the whole sequence
    all_outputs = model.forward(all_input_ids, output_hidden_states=True, return_dict=True)

    # get logits for the last token
    last_token_logits = all_outputs.logits[0][-1:]

    # process the sequence except the last token
    kv_cache = model(all_input_ids[:, :-1]).past_key_values

    # input only the last token with previous kv_cache
    output_with_kvcache = model(all_input_ids[:, -1:], past_key_values=kv_cache, output_hidden_states=True, return_dict=True)
    # extract the last token logits
    new_last_token_logits = output_with_kvcache.logits[0][-1:]

    for layer_idx in range(len(all_outputs.hidden_states)):
        max_diff = torch.abs(
            all_outputs.hidden_states[layer_idx][:, -1, :] - output_with_kvcache.hidden_states[layer_idx]
        ).max()
        max_diffs.setdefault(f"layer {layer_idx}", []).append(max_diff.cpu().item())

    # theese two distributions should be equal, but they are not.
    max_diffs.setdefault("logits", []).append(torch.abs(last_token_logits - new_last_token_logits).max().cpu().item())

for key, value in max_diffs.items():
    print(f"{key}: {sum(value) / len(value)}")
