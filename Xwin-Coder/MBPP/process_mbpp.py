from human_eval.data import stream_jsonl
import glob 
from tqdm import tqdm
import argparse
import jsonlines
import json
from datasets import load_dataset

def read_mbpp():
    mbpp_problems = {}
    ds = list(load_dataset("mbpp")["test"])
    for obj in ds:
        mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems

parser = argparse.ArgumentParser()

# Inputs
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


files = sorted(glob.glob(args.path + '/*.jsonl'))
print("{} files in {}".format(len(files), args.path))

problems = read_mbpp()
output = [[] for _ in range(len(problems))]

for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    if args.add_prompt: 
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            if '```python' in completion: 
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except:
                    pass
            if "__name__ == \"__main__\"" in completion:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()
            
            if "# Example usage" in completion:
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            
            if "# Test examples" in completion:
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            
            output[task_id-11].append(completion)
    
print("save to {}".format(args.out_path))

with open(args.out_path, "w", encoding="utf-8") as fout:
    json.dump(output, fout)