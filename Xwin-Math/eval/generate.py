# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Written by Weiqi Wang
# ---------------------------------------------------------

import json
import os
from typing import Optional, Tuple

import fire
from vllm import LLM, SamplingParams

def generate(model_path: str, dataset_path: str, prompt_path: str = "prompt/xwin-math.json", output_path: Optional[str] = None, tensor_parallel_size: int = 1, test_part: Optional[Tuple[int, int]] = None)-> None:

    """
    Generate model's response using vLLM. 

    Parameters:
        model_path: Path to your model
        dataset_path: Path to your dataset. We provide some data in eval/input as examples. If you wat to use your own dataset, please convert the dataset to json format with a list of test samples. Each sample contains at least "answer" and "type". 
        prompt_path: Path to your prompt. We provide the prompt for Xwin-Math in eval/prompt/xwin-math.json. If you wat to use your own prompt, please convert the prompt to json format with key as the name of the prompt and value as the prompt. 
        output_path: Path to save the results, if left None, the results will be saved in eval/response/{model_name}_{dataset_name}.json
        tensor_parallel_size: The number of GPUs you want to use. 
        test_part: The range of data to be processed. [low, high]. If left None, will process all the data. 
    """

    # Load test data
    with open(dataset_path, 'r') as f:
        test_dataset_list = json.load(f)

    # Load prompt
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)
        
    if test_part is not None:
        test_dataset_list = test_dataset_list[test_part[0]:test_part[1]]
    batch_size = len(test_dataset_list)

    for items in test_dataset_list:
        items["completion"] = ["" for _ in range(len(prompt_dict))]

    llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size)
    stop_tokens = ["</s>", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)

    for index, (_, prompt) in enumerate(prompt_dict.items()):
        for index_global in range(len(test_dataset_list) // batch_size):
            # Batch Construct
            if batch_size *(index_global+2) > len(test_dataset_list):
                curr_data_list = test_dataset_list[batch_size * index_global:]
            else:
                curr_data_list = test_dataset_list[batch_size * index_global:batch_size * (index_global+1)]

            # Inference
            prompt = [prompt.format(instruction = item["question"], ANSWER = "{ANSWER}", NUMBER = "{NUMBER}") for item in curr_data_list]
            print(prompt[0])
            completions = llm.generate(prompt, sampling_params)

            # Save
            for index_in_curr_batch, output in enumerate(completions):
                generated_text = output.outputs[0].text
                index_in_data_list = index_global * batch_size + index_in_curr_batch
                item = test_dataset_list[index_in_data_list]
                item["completion"][index] = generated_text

    if output_path is None:
        output_path = os.path.join("response", f"{os.path.split(model_path)[-1]}_{os.path.split(dataset_path)[-1]}")
        if not os.path.exists("response"):
            os.makedirs("response")

    with open(output_path, 'w') as f:
        json.dump(test_dataset_list, f, indent=4)

if __name__ == "__main__":
    fire.Fire(generate)