# Evaluating APPS

![apps](../docs/apps.png)

## Introduction

The APPS dataset consists of problems collected from different open-access coding websites such as Codeforces, Kattis, and more. The APPS benchmark attempts to mirror how humans programmers are evaluated by posing coding problems in unrestricted natural language and evaluating the correctness of solutions. The problems range in difficulty from introductory to collegiate competition level and measure coding ability as well as problem-solving.[[1]](https://github.com/hendrycks/apps)

Our models reached new SOTA results over all open source models on APPS-introductory, and surpass GPT-3.5-turbo:

| Model | Size | Pass@ | Introductory |  Interview  | Competition |
|----|----|----|----|----|----|
| AlphaCode[[2]](https://arxiv.org/pdf/2203.07814.pdf) | 1B  |  5 (filter 50k) | 20.4  | 9.7  | 7.8  |
|  |   | 1000  | 17.7  | 5.2  | 7.1  |
| GPT-Neo[[3]](https://paperswithcode.com/sota/code-generation-on-apps) | 2.7B  | 5  | 5.5  | 0.8  | 0.0  |
| GPT-J[[3]](https://paperswithcode.com/sota/code-generation-on-apps) | 6B | 5  | 9.2  | 1.7  |  1.0 |
| CodeLLaMA[[2]](https://arxiv.org/pdf/2308.12950.pdf) | 34B  | 5 | 32.8 |  8.8 | 2.9  |
|  |   | 10 | 39.0 | 12.2 | 4.7  |
|   |   | 100 | 56.3 | 24.3 | 15.4  |
| Codex[[4]](https://arxiv.org/pdf/2107.03374.pdf) | 12B  | 5  | 9.7 | 0.5  | 0.1  |
|  |   | 1000  | 25.0 | 3.7  | 3.2  |
| Code-davinci-002[[3]](https://paperswithcode.com/sota/code-generation-on-apps) | - |  1 |  29.3 | 6.4  | 2.5 |
| GPT-3.5-turbo | - |  1 | 25.9  | 9.5  | 3.7 |
| GPT-4 | - |  1 | **60.0**  | **35.1** | **10.2** |
| **XwinCoder**  | 7B  | 5 | 31.5  | 7.9   | 2.9  |
| **XwinCoder** | 13B  | 5 | 35.4  | 11.0  | 3.8 |
|  **XwinCoder** | 34B | 1  | 30.1  |  7.8 | 1.9  |
|   |  | 5  | **43.0** |  **14.1** |  **5.1** |


## How to Evaluate


First download the data:
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
tar -zxvf APPS.tar.gz
rm -f APPS.tar.gz
```
Then run:
```bash
bash eval_apps.sh
```
This will generate $gpu splits of problem and the execution results. To summarize and visualize the final results, run:
```bash
bash merge_and_visual.sh
```

## Further information
The results of GPT-3.5-turbo and GPT-4 are evaluated by prompting the model to "Answer the following code contest problem and return a fully runnable code: ". If you need those results, please kindly cite this repo.