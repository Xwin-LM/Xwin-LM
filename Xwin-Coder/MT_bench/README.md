# Evaluating MT-Bench

## Introduction

The evaluation code of mt-bench is extracted from the [fastchat]() repository. To evaluate our model, you will need a openai GPT-4 API key. Set your model name, api information in `eval.sh`.

Run:
```bash
eval.sh
```
This will return a table like:

![mtbench](../docs/mtbench.png)