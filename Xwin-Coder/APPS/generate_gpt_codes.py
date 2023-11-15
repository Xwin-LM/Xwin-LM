"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from reindent import run as run_reindent

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm
try:
    from vllm import LLM, SamplingParams
except:
    print("vllm not installed, connot add --vllm tag")


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = """<system>: You are an AI coding assistant that helps people with programming. Write a response that appropriately completes the user's request.
<user>: Answer the following code contest problem and return a fully runable code:
"""
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\n<AI>: "

    if args.peeking > 0.0:
        # Need to do some peeking. 

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering
    
    # convert ratio to number
    args.start = int(args.start_ratio*len(problems))
    args.end = int(args.end_ratio*len(problems))
    

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    
    # Only do the problems that are specified.
    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.arch)

    # Set up model
    print("Loading model...")
    if args.vllm:
        model = LLM(model=args.load, dtype = "float16", max_model_len=8192)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.load)
        model.cuda()
    print(f"Loaded {args.load}.")

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")

        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
                starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            continue

        # Read the question in
        prompt_text, sample_sol = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)
        
        # Feed this into the model.
        start = time.time()
        try:
            with torch.no_grad():
                if args.vllm:
                    sampling_params = SamplingParams(
                        use_beam_search=args.num_beams>0,
                        early_stopping=False,
                        best_of=args.num_beams if args.num_beams>0 else args.N,
                        temperature=args.temperature,
                        top_p=0.95 if args.temperature>0 else 1.0,
                        max_tokens=1024,
                        n=args.N,
                    )
                    outputs = model.generate([prompt_text], sampling_params, use_tqdm=False)
                    output_str = [output.text for output in outputs[0].outputs]
                else:
                    input_ids = torch.LongTensor(tokenizer.encode(prompt_text, verbose=False)).unsqueeze(0).cuda()
                    output_ids = model.generate(
                        input_ids,
                        num_beams=args.num_beams,
                        early_stopping=True if args.num_beams>1 else False,
                        do_sample=True if args.temperature>0 else False,
                        temperature=args.temperature,
                        top_p=0.95 if args.temperature>0 else 1.0,
                        max_length=1024 - len(input_ids)
                    )
                    output_str = tokenizer.decode(output_ids[0])
        except Exception as e:
            if isinstance(e, UnboundLocalError) and str(e) == "local variable 'next_tokens' referenced before assignment":
                # See https://github.com/huggingface/transformers/issues/5118
                if args.debug:
                    print("Problem text was > 1024 tokens, so cannot do generation")
                    print(e)
            else:
                print("Unexpected exception in generating solution")
                print(e)
            # Default to empty string on errors
            output_str = ["" for _ in range(args.N)]
        end = time.time()

        if args.peeking == 1.0:
            output_str = sample_sol
        elif len(output_str):
            if args.vllm:
                for i in range(len(output_str)):
                    if "```" in output_str[i]:
                        output_str[i] = output_str[i].split("```")[1]
                        if output_str[i].startswith("python\n"):
                            output_str[i] = output_str[i].replace("python\n", "")
            else:
                for i in range(len(output_str)):
                    if "<AI>:" in output_str[i]:
                        output_str[i] = output_str[i].split("<AI>:")[1]
                    if "```" in output_str[i]:
                        output_str[i] = output_str[i].split("```")[1]
                        if output_str[i].startswith("python\n"):
                            output_str[i] = output_str[i].replace("python\n", "")

        # Save the generated sol
        gpt_codes[index+args.start] = output_str

        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string:")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("-t","--test_loc", default="eval/test_path.json", type=str)
    parser.add_argument("-r","--root", default="./", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", default="~/apps/models/checkpoints/final", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("-s","--start_ratio", default=0.0, type=float)
    parser.add_argument("-e","--end_ratio", default=1.0, type=float)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--vllm", action="store_true", default=False)
    parser.add_argument("--N", default=1, type=int)
    args = parser.parse_args()

    main(args)
