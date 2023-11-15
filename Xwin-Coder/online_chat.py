import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
from rich import print as rprint
from rich.panel import Panel  
from rich.rule import Rule 
from rich.style import Style


class OnlineChat():
    def __init__(self, args):
        self.model = LLM(model=args.model, dtype = "float16", max_model_len=8192)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.prompt_type = args.prompt_type
        self.para = SamplingParams(temperature=args.temperature,
                max_tokens=args.max_len,
                n=1,
                top_p=0.95 if not args.temperature==0 else 1,
                stop=[self.tokenizer.eos_token])
        self.hard_stop = ["<user>", "<AI>", "<system>"]
        self.history = []
    
    def generate(self):
        inp = self.conv_to_prompt()
        outputs = self.model.generate([inp], self.para, use_tqdm=False)[0]
        gen_seqs = [output.text for output in outputs.outputs][0]
        for st in self.hard_stop:
            if st in gen_seqs:
                gen_seqs = gen_seqs.split(st)[0]
        return gen_seqs
    
    def conv_to_prompt(self):
        prompt = "<system>: You are an AI coding assistant that helps people with programming. Write a response that appropriately completes the user's request.\n"
        prefix = {
            "user": "<user>: {content}\n",
            "model": "<AI>: {content}\n"
        }
        for r, conv in self.history:
            prompt = prompt + prefix[r].format(content=conv)
        assert r == "user"
        prompt = prompt + "<AI>: "
        return prompt

    
    def chat_one_turn(self):
        text = input("Input: \n")
        if text == "Q":
            return False, "Q"
        if text == "C":
            return False, "C"
        self.history.append(("user", text))
        try:
            res = self.generate()
        except Exception as e:
            print(f"We cannot continue to generate because{e}, press enter and we will clear chat")
            input()
            return False, "C"
        self.history.append(("model", res))
        return True, ""
    
    def visual(self):
        os.system("clear")
        rprint("Begin chat. \n >>> If you want to quit, input \"Q\"; \n >>> If you want to clear cache and start new chat, input \"C\";\n")
        # visual conversation
        for r, txt in self.history:
            panel = Panel(txt, title=r, style=Style(color="white" if r=="model" else "yellow"))
            rprint(panel)
        rprint(Rule(characters="-"))


    def chat(self):
        self.history = []
        can_continue = True
        while can_continue:
            can_continue, msg = self.chat_one_turn()
            self.visual()
        return True if msg == "C" else False

    def loop(self):
        chat = True
        while chat:
            os.system("clear")
            rprint("Begin chat. \n >>> If you want to quit, input \"Q\"; \n >>> If you want to clear cache and start new chat, input \"C\";\n")
            chat = self.chat()  # return false only Q





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--max_len', type=int, default=2048, help="")
    parser.add_argument('--prompt-type', type=str, default='Author', help="")

    args = parser.parse_args()

    chat = OnlineChat(args)
    chat.loop()

if __name__ == "__main__":
    main()