import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
from human_eval.data import write_jsonl
from ds1000ds import DS1000Dataset
import itertools
try:
    from vllm import LLM, SamplingParams
except:
    print("vllm not installed, connot add --vllm tag")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

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
        redataset[problem_id] = d["prompt"]
    return redataset


def change_format(txt):
    tmptxt = deepcopy(txt)
    
    if "\nA:\n<code>" in tmptxt:
        prompt, code = tmptxt.split("\nA:\n<code>")
    else:
        prompt = tmptxt
        code = ""
    require = None
    if "<code>" in code:
        stid = tmptxt.index("<code>")
        assert "</code>" in code[:stid], code
        precode = code[: code.index("</code>")]
        require = code[code.index("</code>")+len("</code>"): stid]
    else:
        assert "</code>" not in code
        precode = code

    if require:
        prompt = prompt + "You should write codes ending with: \n" + require
    return prompt, precode
        
def generate_prompt(input):
    pattern = re.compile(r"A:.*?<code>", re.DOTALL) 
    QAL = re.split(pattern, input)
    if len(QAL)==1:
        return f"""<system>: You are an AI coding assistant that helps people with programming. Write a response that appropriately completes the user's request.
<user>: Complete the task in the comment and return a fully runable code.
{input}
<AI>: Here is the solution: ```python
{input}
"""
    else:
        assert len(QAL) == 2
        prompt, code = QAL
        if "<code>" in code:
            precode, ed = code.split("<code>") # the second <code> for model to start
            assert "</code>" in precode, input
            assert ed.replace("\n", "").strip() == ""
            precode, st_mark = precode.split("</code>")
            req = st_mark.replace("BEGIN SOLUTION", "")
            prompt = prompt + f"\nYour solution code should ends with: {req}"
        else:
            assert "</code>" not in code
            precode = code
        return  f"""<system>: You are an AI coding assistant that helps people with programming. Write a response that appropriately completes the user's request.
<user>: {prompt}
<AI>: Here is the solution code of this problem: ```python
{precode}
"""


def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
    page_attention: bool = False
):
    config = AutoConfig.from_pretrained(base_model)
    print(config)
    if not page_attention:
        assert base_model, (
            "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        model.config.pad_token_id = tokenizer.pad_token_id

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        return tokenizer, model
    else:
        assert base_model, (
            "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
        )
        assert not load_8bit, NotImplementedError
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = LLM(model=base_model, dtype = "float16")
        return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--subset', type=str, default=None, help="One of competition, interview, introductory")

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--vllm', action='store_true', default = False, help='')
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_ratio', type=float, default=0, help="")
    parser.add_argument('--end_ratio', type=float, default=1, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()

    # assert args.subset in ["All", "Pandas"], "error subset name"
    

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = read_problems(args.subset)


    task_ids = list(sorted(problems.keys()))
    args.start_index = int(args.start_ratio*len(task_ids))
    args.end_index = int(args.end_ratio*len(task_ids))
    task_ids = task_ids[args.start_index: args.end_index]
    prompts = [problems[task_id] for task_id in task_ids]

    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model, page_attention=args.vllm)
    

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt)]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        if not args.vllm:
            generation_config = GenerationConfig(
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False if args.temperature==0.0 else True,
                temperature=args.temperature,
                max_length=args.max_len,
                num_return_sequences=args.num_seqs_per_iter,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95
            )
            encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

            if args.decoding_style == 'sampling':
                loops = int(args.N / args.num_seqs_per_iter)
            else:
                loops = 1

            for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

                with torch.no_grad():
                    gen_tokens = model.generate(
                        **encoding,
                        generation_config=generation_config
                    )

                if gen_tokens is not None:
                    gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                else:
                    gen_seqs = None

                if gen_seqs is not None:
                    assert len(ids_batch) == 1
                    task_id = ids_batch[0]

                    for seq_idx, gen_seq in enumerate(gen_seqs):
                        completion_seq = gen_seq
                        
                        if gen_seq.startswith(prompt_batch[0]):
                            completion_seq = gen_seq.replace(prompt_batch[0], "")

                        completion_seq = completion_seq.replace('\t', '    ')
                        all_code = gen_seq.replace('\t', '    ')

                        completion_seqs.append(
                            {'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                            }
                        )
        else:
            
            #prompt_batch: [prompt]
            # encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            sampling_params = SamplingParams(
                temperature=args.temperature,
                frequency_penalty=0.05, 
                max_tokens=args.max_len,
                n=args.num_seqs_per_iter,
                top_p=0.95 if args.temperature>0.0 else 1,
                stop = ["<system>", "<language>", "<AI>", "<code>", "</code>"]
            )

            if args.decoding_style == 'sampling':
                loops = int(args.N / args.num_seqs_per_iter)
            else:
                loops = 1

            for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
                
                outputs = model.generate(prompt_batch, sampling_params, use_tqdm=False)[0]
                gen_seqs = [output.text for output in outputs.outputs]

                if gen_seqs is not None:
                    assert len(ids_batch) == 1
                    task_id = ids_batch[0]

                    for seq_idx, gen_seq in enumerate(gen_seqs):
                        completion_seq = gen_seq

                        completion_seq = completion_seq.replace('\t', '    ')
                        all_code = gen_seq.replace('\t', '    ')

                        completion_seqs.append(
                            {'task_id': task_id,
                            'completion': completion_seq,
                            'all_code': all_code,
                            }
                        )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
