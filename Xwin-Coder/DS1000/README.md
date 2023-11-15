# Evaluating DS1000

![ds1000](../docs/ds1000.jpeg)

## Introduction

DS-1000 is a code generation benchmark with a thousand data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.[[1]](https://ds1000-code-gen.github.io/)

Here are the results of our models. **We only state that our models show promising ability to help users to gain knowledge  about common python packages.** Because DS1000 was designed for pretrained model (only complete sentences), and we use additional tricks to evaluate our model, and there are issues of testing GPT-3.5-turbo's ability (see 'Further information' below). So the comparison on this dataset is not fully fair.


| Model | Size | Matplotlib | Numpy | Pandas | Pytorch | Scipy | Sklearn | Tensorflow | Overall |
|----|----|----|----|----|----|----|----|----|----|
|# of Problems|   | 155 | 220 | 291 | 68  |  106  |  115 | 45 | 1000 |
|----------------------|----|------------|--------|--------|---------|------|--------|-------------|--------|
|Code-Cushman-001[[2]](https://arxiv.org/pdf/2306.08568.pdf)| Nan | 40.7 | 21.8 | 7.9  | 12.4 |  11.3  | 18.0 | 12.2 | 18.1 |
| StarCoder[[3]](https://arxiv.org/pdf/2305.06161.pdf) | 15B | 51.7 | 29.7 | 11.4  | 21.4 |  20.2  | 29.5 | 24.5 | 26.0 |
| WizardCoder[[2]](https://arxiv.org/pdf/2306.08568.pdf) | 15B | 55.2 | 33.6 | 16.7  | 26.2 |  24.2  | 24.9 | 26.7 | 29.2 |
| Code-davinci-002[[4]](https://github.com/xlang-ai/DS-1000) | Nan | 55.5 | 44.6 | 26.5  | 39.7 |  32.1  | 46.1 | 40.0 | 39.3 |
| GPT-3.5-turbo | Nan | 6.45(fail) | 30.0 |  19.24   | 17.65  |  19.81   |  19.13 | 33.33  | 20.2   |
| GPT-4 | Nan | 65.16 | 60.0  |  46.74  |  38.24 |  42.45   |  55.65 | 51.11  | 52.7  |
|----------------------|----|------------|--------|--------|---------|------|--------|-------------|--------|
| **XwinCoder** | 7B |  46.5  |  32.7  | 19.9 |  22.1  | 17.9 | 20.9 | 22.2 | 27.0 |
| **XwinCoder** | 13B | 61.9 | 42.3 | 21.7 | 38.2 | 26.4 | 31.3 | 26.7 | 35.4 |
| **XwinCoder** | 33B | 58.71 | 48.64  | 29.21 | 39.71 | 36.79  | 45.22 | 24.44 | 41.2 |


## How to Evaluate

The original DS1000 dataset was designed for pretrained models. However, after instruction finetuning, the performance of directly using original dataset's prompt and continue to generate drop to very low. This may be because the complicts of system prompt formats. We evaluate the true ability of instruction finetuned model by hacking the prompt format into ours, i.e., change `A:` in origianl prompts into ours `<AI>: ` and change `<code>` into " ``` ".

### 1. Generate Responses
Download data and unzip from original [repository](https://github.com/HKUNLP/DS-1000/blob/main/ds1000_data.zip) and unzip it. Then set model name and run:
```bash
bash generate_ds1000.sh
```
### 2. Check Correctness
Evaluating using [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main) requires a lower version of python. Instead, using the execution of original code is also fast and convenient. We suggest start a new docker and install DS1000 requirements and then run the evaluation scripts
```bash
sudo docker run -it -p 8022:22 -d --name=<docker name> --privileged --net=host --ipc=host --gpus=all -v /:/data superbench/dev:cuda11.8 bash
sudo docker exec -it <docker name> bash
cd /data/path/to/Xwin-LM/Xwin-Coder/DS1000
pip install -r ds1000_requirements.txt
bash eval_ds1000.sh
```


## Further information

#### 1. Why hacking the prompt to evaluate?

DS1000 was designed for pretrained models (or sentence completion models). Instruction finetuned models tend to return a complete code, including import, original context code, solution part, example usages and etc. There are no general rules that can extract only the solution part out of this response.  

#### 2. How we evaluate GPT-4 and GPT-3.5-turbo?

The results of GPT4 and GPT-3.5-turbo is evaluated by prompting the model to : "Only continue after the following prompt: ". (that is, we want the chat model to act like a pretrained model). Because we can not hack the prompt format of openai's models. 

#### 3. Why GPT-3.5-turbo is so weak on this benchmark? How fair is the comparison?

Manually checking the genration results, GPT-4 succeeded to follow the instruction (to only continue the prompt) for all examples, which means no unnecessary errors are introduced. We believe its results have reference significance.

GPT-3.5-turbo, on the contrary, failed to do so sometimes. Thus mistakes that not relevant to coding ability are introduced. It is not fair to say GPT-3.5-turbo is worse than any other models (except GPT-4).

**However, we believe that DS1000 is a benchmark that makes great engineering sense in the real world. If anyone figure out how to evaluate GPT-3.5-turbo on this benchmark well, we are very willing to hear and update our results.**