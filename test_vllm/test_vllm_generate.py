# SPDX-License-Identifier: Apache-2.0
import vllm
from vllm import LLM, SamplingParams
import argparse

def inference(base_model_path:str):
    # Sample prompts.
    prompts = [
        "你好，我是一名ML算法工程师，请介绍一下vllm库",
        "降压药有哪些?分别有什么优缺点",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model=base_model_path, trust_remote_code=True)
    print(f"llm model:{llm}")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', default="~/data/work/hf_data_and_model/models/Qwen/Qwen2.5-3B/", type=str, help='')
    # 添加一个参数来捕获剩余的所有参数
    parser.add_argument("unknown_args", nargs=argparse.REMAINDER, help="Unknown arguments")

    model_args = parser.parse_args()
    print(model_args)
    #print(unknown_args)
    inference(model_args.base_model_path)