from human_eval.data import stream_jsonl
import glob 
from tqdm import tqdm
import argparse
import jsonlines
import json
from ds1000ds import DS1000Dataset
import itertools
import os 

def read_problems(subset):
    redataset = {}
    data = DS1000Dataset("ds1000_data", mode="Completion").data  
    if subset == "All":  
        dataset = list(itertools.chain(*data.values()))  
    else:  
        dataset = data[subset]  

    # Add dataset to redataset  
    for i, d in enumerate(dataset):  
        problem_id = i 
        redataset[problem_id] = d
    return redataset

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--subset', type=str, default=None, help="One of competition, interview, introductory")


parser.add_argument(
    '--path',
    type=str,
    help="")
parser.add_argument(
    '--out_path',
    type=str,
    help="")
parser.add_argument(
    '--add_prompt',
    action='store_true',
    help='')

args = parser.parse_args()

# assert args.subset in ["All", "Pandas"], "error subset name"

files = sorted(glob.glob(args.path + '/*.jsonl'))
print("{} files in {}".format(len(files), args.path))

problems = read_problems(args.subset)
output = [[] for _ in range(len(problems))]
a = 0
correct = 0
n_samples = len(problems)
for idx in tqdm(range(len(files))):
    file_path = os.path.join(args.path, f"{idx}.jsonl")  
    codes = [c for c in list(jsonlines.open(file_path))] 
    if args.add_prompt: 
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']

            if '```' in completion: 
                completion = completion.split('```')[0]
            
            if "# Example usage" in completion:
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            
            if "# Test examples" in completion:
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            
            correct += int(problems[task_id].test(completion))
            output[idx].append(completion)

acc = correct/n_samples
print(f"correct: {correct}, all: {n_samples}, acc: {acc}")
print("save to {}".format(args.out_path))
print(a)
with open(args.out_path, "w", encoding="utf-8") as fout:
    json.dump(output, fout)