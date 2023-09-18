<div align="center">
  <img src="assets/logo.png" style="width: 70%">
</div>

<h3 align="center">
Powerful, Stable, and Reproducible LLM Alignment
</h3>

<p align="center">
  <a href="https://huggingface.co/Xwin-LM">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue">
  </a>
</p>



**Step up your LLM alignment with Xwin-LM!**

Xwin-LM aims to develop and open-source alignment technologies for large language models, including supervised fine-tuning (SFT), reward models, reject sampling, reinforcement learning, etc. Our first release, built-upon on the Llama2 base models, ranked **TOP-1** on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). Notably, it's **the first to surpass GPT-4** on this benchmark. The project will be continuously updated.

## News

- :boom: [Sep, 2023] We released [Xwin-LM-70B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1), which has achieved a win-rate against Davinci-003 of **95.57%** on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) benchmark, ranking as **TOP-1** on AlpacaEval. **It was the FIRST model surpassing GPT-4** on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). Also note its winrate v.s. GPT-4 is **60.61**.
- :boom: [Sep, 2023] We released [Xwin-LM-13B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-13B-V0.1), which has achieved **91.76%** win-rate on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), ranking as **top-1** among all 13B models.
- :boom: [Sep, 2023] We released [Xwin-LM-7B-V0.1](https://huggingface.co/Xwin-LM/Xwin-LM-7B-V0.1), which has achieved **87.82%** win-rate on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), ranking as **top-1** among all 7B models.


## Model Card
| Model        | Checkpoint | Report | License  |
|------------|------------|-------------|------------------|
|Xwin-LM-7B-V0.1| 🤗 <a href="https://huggingface.co/Xwin-LM/Xwin-LM-7B-V0.1" target="_blank">HF Link</a> | 📃**Coming soon (Stay tuned)** | <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License|
|Xwin-LM-13B-V0.1| 🤗 <a href="https://huggingface.co/Xwin-LM/Xwin-LM-13B-V0.1" target="_blank">HF Link</a> |  |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License|
|Xwin-LM-70B-V0.1| 🤗 <a href="https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1" target="_blank">HF Link</a> |  |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License|
## Benchmarks

### Xwin-LM performance on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/).

The table below displays the performance of Xwin-LM on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), where evaluates its win-rate against Text-Davinci-003 across 805 questions. To provide a comprehensive evaluation, we present, for the first time, the win-rate against ChatGPT and GPT-4 as well. Our Xwin-LM model family establish a new state-of-the-art performance across all metrics. Notably, Xwin-LM-70B-V0.1 has eclipsed GPT-4 for the first time, achieving an impressive win-rate of **95.57%** to Text-Davinci-003 and **60.61%** to GPT-4.

| **Model**                         | **AlpacaEval (winrate %)** | **AlpacaEval (winrate %)**  |**AlpacaEval (winrate %)** |
|----------------------------------|------------|----------|-------------|
|    | **v.s. Text-Davinci-003** | **v.s. ChatGPT** | **v.s. GPT4**|
| **Xwin-LM-70B-V0.1**                   | **95.57**      | **87.50**          | **60.61**            |
| GPT-4                          | 95.28  | 84.66     |  50.00      |
| WizardLM-70B-V1.0           | 92.91  | 80.19     |  46.70      |
| Llama-2-70B-Chat         | 92.66  | 81.97     |  51.19      |
| **Xwin-LM-13B-V0.1**          | **91.76**  | **81.79**  |  **55.30**    |
| ChatGPT                    | 89.37  | 50.00     |  16.60      |
| WizardLM-13B-V1.2         | 89.17  | 75.75     |  41.91      |
| **Xwin-LM-7B-V0.1**      | **87.35** | **76.40**    |  **47.57**     |
| Llama-2-13B-Chat     | 81.09  | 64.22     |  30.92      |

### Case study on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)

|   Prompts  |  GTP4 response   |  Xwin-LM response |
|----------------|------------|--------|
|I've put a plate on top of an apple, and take the plate from kitchen to dinning room. Where is the apple now|The apple is now in the dining room, as it was underneath the plate when you moved the plate from the kitchen to the dining room.|The apple is still in the kitchen, as you have only moved the plate. The apple is under the plate, so if you want to bring the apple to the dining room, you should either move the apple directly or put the apple on the plate and then move the plate to the dining room.|
| Answer the following question.<br> How do you say "good evening" in French. | In French, "good evening" is said as "bonsoir." | In French, "good evening" is translated to "bonsoir." This greeting is used in the evening and nighttime hours, typically after 6 p.m. It is a combination of "bon," which means "good," and "soir," which means "evening." To pronounce it, say "bone-swahr." |


### Xwin-LM performance on NLP foundation tasks.

The following table provides a comparison of Xwin-LMs with other LLMs on NLP foundation tasks in [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

| Model            | MMLU 5-shot | ARC 25-shot | TruthfulQA 0-shot | HellaSwag 10-shot | Average    |
|------------------|-------------|-------------|-------------------|-------------------|------------|
| Text-davinci-003 | 56.9 | 85.2    | 59.3         | 82.2       | 70.9  |
|Vicuna-13b 1.1   | 51.3        | 53.0        | 51.8              | 80.1              | 59.1       |
|Guanaco 30B   | 57.6        | 63.7        | 50.7              | 85.1              | 64.3       |   
| WizardLM-7B 1.0      | 42.7        | 51.6        | 44.7              | 77.7              | 54.2       |
| WizardLM-13B 1.0     | 52.3        | 57.2        | 50.5              | 81.0              | 60.2       |
| WizardLM-30B 1.0    | 58.8    | 62.5 | 52.4       | 83.3          | 64.2|
| Llama-2-7B-Chat      | 48.3        | 52.9        | 45.6     | 78.6    | 56.4       |
| Llama-2-13B-Chat      | 54.6        | 59.0        | 44.1     | 81.9    | 59.9       |
| Llama-2-70B-Chat      | 63.9        | 64.6        | 52.8       | 85.9   | 66.8       |
| **Xwin-LM-7B-V0.1**      | 49.7        | 56.2        | 48.1     | 79.5    | 58.4       |
| **Xwin-LM-13B-V0.1**      | 56.6        | 62.4        | 45.5     | 83.0    | 61.9       |
| **Xwin-LM-70B-V0.1**      | 69.6        | 70.5        | 60.1       | 87.1   | 71.8       |


## Inference

### Conversation templates
To obtain desired results, please strictly follow the conversation templates when utilizing our model for inference. Our model adopts the prompt format established by [Vicuna](https://github.com/lm-sys/FastChat) and is equipped to support **multi-turn** conversations.
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi! ASSISTANT: Hello.</s>USER: Who are you? ASSISTANT: I am Xwin-LM.</s>......
```

### HuggingFace Example

```
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Xwin-LM/Xwin-LM-7B-V0.1")
tokenizer = AutoTokenizer.from_pretrained("Xwin-LM/Xwin-LM-7B-V0.1")
(
    prompt := "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: Hello, can you help me? "
            "ASSISTANT:"
)
inputs = tokenizer(prompt, return_tensors="pt")
samples = model.generate(**inputs, max_new_tokens=4096, temperature=0.7)
output = tokenizer.decode(samples[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(output) 
# Of course! I'm here to help. Please feel free to ask your question or describe the issue you're having, and I'll do my best to assist you.
```


### vllm Example
Because Xwin-LM is based on Llama2, it also offers support for rapid inference using [vllm](https://github.com/vllm-project/vllm). Please refer to [vllm](https://github.com/vllm-project/vllm) for detailed installation instructions.
```
from vllm import LLM, SamplingParams
(
    prompt := "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: Hello, can you help me? "
            "ASSISTANT:"
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)
llm = LLM(model="Xwin-LM/Xwin-LM-7B-V0.1")
outputs = llm.generate([prompt,], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(generated_text)
```

## TODO

- [ ] Release the source code
- [ ] Release more capabilities, such as math, reasoning, and etc.

## Citation
Please consider citing our work if you use the data or code in this repo.
```
@software{xwin-lm,
  title = {Xwin-LM},
  author = {Xwin-LM Team},
  url = {https://github.com/Xwin-LM/Xwin-LM},
  version = {pre-release},
  year = {2023},
  month = {9},
}
```

## Acknowledgements

Thanks to [Llama 2](https://ai.meta.com/llama/), [FastChat](https://github.com/lm-sys/FastChat), [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm), and [vllm](https://github.com/vllm-project/vllm).
