import json
import os
import argparse
from collections import defaultdict

import openai

openai.api_key = "YOUR_API_KEY"

INSTRUCTION = "Write {} instructions with similar meaning as the following prompts: {}. Prompt: =>"
INSTRUCTION_WITH_DEMO = """Write {} instructions with similar meaning as the following prompts:\n{}\nPrompt =>\n{}\n\n
Write {} instructions with similar meaning as the following prompts:\n{}\nPrompt =>"""
DEMO_INSTRUCTIONS = ["Can you suggest an item with a {}-star rating for the user?",
                     "Please lookup a {}-star item recommendation for the user.",
                     "Could you provide a recommendation for a {}-star item to the user, please?",
                     "I need you to find a {}-star item suggestion for the user.",
                     "Please advise the user on a {}-star item that you would recommend."]

PROMPT_ANSWERS = "Please provide a prompt to respond the following request, be precise, no explanation is needed:\n" \
                 "request => Can you suggest an item with a {}-star rating for the user?\n" \
                 "response => We recommend the following item(s) with " \
                 "a {}-star rating for the user: {}.\n\n" \
                 "Please provide a prompt to respond the following request, be precise, no explanation is needed:\n" \
                 "request => <request>\n" \
                 "response => "


def str2bool(str_):
    if str_.lower() in ["1", "yes", "true"]:
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_file", type=str, default="trigger.json")
    parser.add_argument("--dump_file", type=str, default="prompts.json")
    parser.add_argument("--instruction_with_demo", type=str2bool, default="True")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")

    parser.add_argument("--rating_num", type=int, default=100,
                        help="number of prompts generated for rating task")
    parser.add_argument("--sequential_num", type=int, default=100,
                        help="number of prompts generated for sequential task")
    parser.add_argument("--explanation_num", type=int, default=100,
                        help="number of prompts generated for explanation task")
    parser.add_argument("--review_num", type=int, default=10,
                        help="number of prompts generated for review task")
    parser.add_argument("--traditional_num", type=int, default=100,
                        help="number of prompts generated for traditional task")
    parser.add_argument("--item_desc_num", type=int, default=100)
    parser.add_argument("--seq_desc_num", type=int, default=100)

    args = parser.parse_args()
    return args


def call_gpt(prompt):
    message = [{"role": "user", "content": prompt}]
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                              messages=message)
    response = completion['choices'][0]['message']['content']
    return response


class Prompt(object):
    def __init__(self, args, num_prompts: dict[str, int]):
        self.template_file = args.template_file
        self.instruction_with_demo = args.instruction_with_demo
        self.prompt_file = args.dump_file
        self.num_prompts = num_prompts
        self.sep_token = "\n"

        self.prompts = defaultdict(dict)
        self.template = self.load_template()

    def __call__(self):
        self.generate_prompt()

    def load_template(self):
        with open(self.template_file, "r") as fp:
            template = json.load(fp)
        return template

    def generate_prompt(self):
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, "r") as fp:
                self.prompts = json.load(fp)

        if not "info" in self.prompts.keys():
            self.prompts["info"] = {"use_demo": self.instruction_with_demo,
                                    "instruction": INSTRUCTION_WITH_DEMO if self.instruction_with_demo else INSTRUCTION}

        for task, groups in self.template.items():
            print("generating prompt for {} task...".format(task))
            assert task in self.num_prompts.keys()
            for idx, group in groups.items():
                # 如果已经有足够的prompt，则跳过生成
                if task not in self.prompts.keys():
                    self.prompts[task] = {}
                if idx not in self.prompts[task].keys():
                    num_to_be_generated = self.num_prompts[task]
                    self.prompts[task][idx] = {"completion": []}
                else:
                    num_to_be_generated = self.num_prompts[task] - len(self.prompts[task][idx]["completion"])
                if num_to_be_generated <= 0:
                    continue

                temp_str = self.sep_token.format(self.sep_token).join(group)
                if not self.instruction_with_demo:
                    trigger = INSTRUCTION.format(num_to_be_generated, temp_str)
                else:
                    num_demos = len(DEMO_INSTRUCTIONS) - 2
                    demo_str = "{}".format(self.sep_token).join(DEMO_INSTRUCTIONS[:2])
                    tar_demo_str = "\n".join(
                        ["{}. {}".format(i + 1, demo) for i, demo in enumerate(DEMO_INSTRUCTIONS[2:])])
                    trigger = INSTRUCTION_WITH_DEMO.format(num_demos,
                                                           demo_str,
                                                           tar_demo_str,
                                                           num_to_be_generated,
                                                           temp_str)
                print(trigger + "\n")
                prompt = call_gpt(trigger)
                if "trigger" not in self.prompts[task][idx].keys():
                    self.prompts[task][idx]["trigger"] = group
                self.prompts[task][idx]["completion"].extend(prompt.split("\n"))

        with open(self.prompt_file, "w") as fp:
            json.dump(self.prompts, fp)


if __name__ == '__main__':
    args = parse_args()
    prompt = Prompt(args,
                    {"rating": args.rating_num,
                     "sequential": args.sequential_num,
                     "explanation": args.explanation_num,
                     "review": args.review_num,
                     "traditional": args.traditional_num,
                     "item_desc": args.item_desc_num,
                     "seq_desc": args.seq_desc_num}
                    )
    prompt()
