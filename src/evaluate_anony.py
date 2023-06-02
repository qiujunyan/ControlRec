import argparse
import os
import random
import sys
import warnings
from pathlib import Path

project_dir = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(project_dir)

from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from models import ControlRecPretraining, P5Pretraining
from notebooks.evaluate.metrics4rec import evaluate_all
from notebooks.evaluate.utils import rouge_score, bleu_score, root_mean_square_error, \
    mean_absolute_error
from param import parse_args
from pretrain import TrainerBase
from pretrain_data import get_loader
from tokenization import P5Tokenizer
from utils import *

warnings.filterwarnings("ignore")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PretrainedModel(TrainerBase):
    def __init__(self, args, model_class):
        super().__init__(args)
        self.args = args
        if 't5' in args.backbone:
            tokenizer = P5Tokenizer.from_pretrained(
                args.backbone,
                model_max_length=args.max_input_length,
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir,
                truncation=True,
                local_files_only=args.local_files_only)
        else:
            raise NotImplementedError()
        self.tokenizer = tokenizer
        config = self.create_config()
        self.model = self.create_model(model_class=model_class, config=config)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer
        self.load_checkpoint(self.args.ckpt_path)


class Evaluator(object):
    def __init__(self, args, model, model_name):
        self.args = args
        self.model = model
        self.tokenizer = model.tokenizer
        self.sample_numbers = {'rating': 1, 'sequential': (1, 1, 1, 1), 'explanation': 1, 'review': 1,
                               'traditional': (1, 1, 1)}

        self.result = {}
        self.log_file = os.path.join(args.eval_dir, model_name + ".log")
        if not os.path.isdir(args.eval_dir):
            os.makedirs(args.eval_dir, exist_ok=True)
        if os.path.exists(self.log_file) and self.args.local_rank == 0:
            os.remove(self.log_file)
        if self.args.local_rank == 0:
            logging.info(f"Log file: {self.log_file}")

    def write_eval_result(self, task, log):
        if self.args.local_rank != 0:
            return
        with open(self.log_file, "a") as fp:
            fp.write(task + ":\n")
            for key, val in log.items():
                fp.write("{}:\t{}\n".format(key, val))
            fp.write("\n")


class P5Evaluator(Evaluator):
    def __init__(self, args, model, model_name):
        super().__init__(args, model, model_name)
        self.prompts = self.load_prompts()

    def load_prompts(self):
        if self.args.trigger_as_prompt:
            with open("prompt/trigger.json", "r") as fp:
                template = json.load(fp)
        else:
            with open("prompt/promptsV2.json", "r") as fp:
                template = {}
                contents = json.load(fp)
                for task_name, groups in contents.items():
                    if task_name == "info":
                        continue
                    task_prompt = {}
                    for group_id, group in groups.items():
                        task_prompt[group_id] = [completion[len(f"{i}. "):] for i, completion
                                                 in enumerate(group["trigger"])]
                    template[task_name] = task_prompt
        return template

    def encode(self, prompt_ids, slot_ids):
        # encode prompt ids
        prompt_attention_mask = torch.ne(prompt_ids, self.tokenizer.pad_token_id)
        prompt_attention_outputs = self.model.encoder(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask
        )
        prompt_hidden_states = prompt_attention_outputs.last_hidden_state

        # encode slot ids
        slot_attention_mask = torch.ne(slot_ids, self.tokenizer.pad_token_id)
        slot_encoder_outputs = self.model.encoder(
            input_ids=slot_ids,
            attention_mask=slot_attention_mask,
        )
        slot_hidden_states = slot_encoder_outputs.last_hidden_state

        attention_mask = torch.cat([prompt_attention_mask, slot_attention_mask], 1)
        output_hidden_states = torch.cat([prompt_hidden_states, slot_hidden_states], 1)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=output_hidden_states), attention_mask

    def eval_rating(self, test_task_list=None, test_sample_numbers=None):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"rating": ["1-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")
        gt_ratings = []
        pred_ratings = []
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                results = self.model.generate_step(batch)
                gt_ratings.extend(batch['target_text'])
                pred_ratings.extend(results)
            break

        predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
                            p in [str(i / 10.0) for i in list(range(10, 50))]]
        RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
        MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
        self.write_eval_result("sample_num: {}\nrating: ".format(len(dataloader)),
                               {"RMSE": 'RMSE {:7.4f}'.format(RMSE), "MAE": 'MAE {:7.4f}'.format(MAE)})

    def eval_sequential(self):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"sequential": ["2-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")
        all_info = []
        num_beams = 2
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.model.device)
                results = self.model.generate_step(batch)
                beam_outputs = self.model.generate(
                    input_ids,
                    max_length=16,
                    num_beams=num_beams,
                    no_repeat_ngram_size=0,
                    num_return_sequences=num_beams,
                    early_stopping=True
                )
                generated_sents = self.model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                for j, item in enumerate(zip(results, batch['target_text'])):
                    new_info = {}
                    new_info['target_item'] = item[1]
                    new_info['gen_item_list'] = generated_sents[j * num_beams: (j + 1) * num_beams]
                    all_info.append(new_info)
            break

        gt = {}
        ui_scores = {}
        for i, info in enumerate(all_info):
            gt[i] = [int(info['target_item'][len("item_"):])]
            pred_dict = {}
            for j in range(len(info['gen_item_list'])):
                try:
                    pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
                except:
                    pred_dict[0] = -(j + 1)
                    pass
            ui_scores[i] = pred_dict

        msg_5, res_5 = evaluate_all(ui_scores, gt, 5)
        msg_10, res_10 = evaluate_all(ui_scores, gt, 10)
        self.write_eval_result("sample_num:{}\nsequential@5: ".format(len(dataloader)), res_5)
        self.write_eval_result("sequential@10: ", res_10)

    def eval_explanation(self):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"explanation": ["3-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")

        tokens_predict = []
        tokens_test = []

        for i, batch in tqdm(enumerate(dataloader), desc="explanation"):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.model.device)
                outputs = self.model.generate(
                    input_ids,
                    min_length=9,
                    num_beams=12,
                    num_return_sequences=1,
                    num_beam_groups=3,
                    repetition_penalty=0.7
                )
                results = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                tokens_predict.extend(results)
                tokens_test.extend(batch['target_text'])
            break

        new_tokens_predict = [l.split() for l in tokens_predict]
        new_tokens_test = [ll.split() for ll in tokens_test]
        BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
        BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
        ROUGE = rouge_score(tokens_test, tokens_predict)

        result = {"BLEU-1": "{:7.4f}".format(BLEU1),
                  "BLEU-4": "{:7.4f}".format(BLEU4)}
        for (k, v) in ROUGE.items():
            result[k] = "{:7.4f}".format(v)
        self.write_eval_result("sample_num:{}\nexplanation".format(len(dataloader)), result)

    def eval_review(self):
        """Since T0 & GPT-2 checkpoints hosted on Hugging Face platform are slow to conduct inference,
        we only perform evaluation on the first 800 instances for prompts in Task Family 4."""
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"review": ["4-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")
        gt_ratings = []
        pred_ratings = []
        for i, batch in tqdm(enumerate(dataloader), desc="review"):
            if i > 50:
                break
            with torch.no_grad():
                results = self.model.generate_step(batch)
                gt_ratings.extend(batch['target_text'])
                pred_ratings.extend(results)

        pred_ratings = [p if is_number(p) else "0.0" for p in pred_ratings]
        predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
        RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
        MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
        self.write_eval_result("sample_num:{}\nreview".format(len(dataloader)),
                               {"RMSE": "{:7.4f}".format(RMSE), "MAE": "{:7.4f}".format(MAE)})

    def eval_traditional(self):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"traditional": ["5-2"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")

        all_info = []
        num_beams = 2
        for i, batch in tqdm(enumerate(dataloader), desc="traditional"):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.model.device)

                results = self.model.generate_step(batch)
                beam_outputs = self.model.generate(
                    input_ids,
                    max_length=50,
                    num_beams=num_beams,
                    no_repeat_ngram_size=0,
                    num_return_sequences=num_beams,
                    early_stopping=True
                )
                generated_sents = self.model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                for j, item in enumerate(zip(results, batch['target_text'])):
                    new_info = {}
                    new_info['target_item'] = item[1]
                    new_info['gen_item_list'] = generated_sents[j * num_beams: (j + 1) * num_beams]
                    all_info.append(new_info)
            break

        gt = {}
        ui_scores = {}
        for i, info in enumerate(all_info):
            gt[i] = [int(info['target_item'][len("item_"):])]
            pred_dict = {}
            for j in range(len(info['gen_item_list'])):
                try:
                    pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
                except:
                    pred_dict[0] = -(j + 1)
            ui_scores[i] = pred_dict

        msg_1, res_1 = evaluate_all(ui_scores, gt, 1)
        msg_5, res_5 = evaluate_all(ui_scores, gt, 5)
        msg_10, res_10 = evaluate_all(ui_scores, gt, 10)
        self.write_eval_result("sample_num:{}\ntraditional@1: ".format(len(dataloader)), res_1)
        self.write_eval_result("traditional@5: ", res_5)
        self.write_eval_result("traditional@10: ", res_10)


class CPLRecSysEvaluator(Evaluator):
    def __init__(self, args, model, model_name):
        super().__init__(args, model, model_name)
        self.prompts = self.load_prompts()

    def load_prompts(self):
        if self.args.trigger_as_prompt:
            with open("prompt/trigger.json", "r") as fp:
                template = json.load(fp)
        else:
            with open("prompt/promptsV2.json", "r") as fp:
                template = {}
                contents = json.load(fp)
                for task_name, groups in contents.items():
                    if task_name == "info":
                        continue
                    task_prompt = {}
                    for group_id, group in groups.items():
                        task_prompt[group_id] = [completion[len(f"{i}. "):] for i, completion
                                                 in enumerate(group["trigger"])]
                    template[task_name] = task_prompt
        return template

    def encode(self, prompt_ids, slot_ids):
        # encode prompt ids
        prompt_attention_mask = torch.ne(prompt_ids, self.tokenizer.pad_token_id)
        prompt_attention_outputs = self.model.encoder(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            return_dict=True
        )
        prompt_hidden_states = prompt_attention_outputs.last_hidden_state

        # encode slot ids
        slot_attention_mask = torch.ne(slot_ids, self.tokenizer.pad_token_id)
        slot_encoder_outputs = self.model.encoder(
            input_ids=slot_ids,
            attention_mask=slot_attention_mask,
            return_dict=True
        )
        slot_hidden_states = slot_encoder_outputs.last_hidden_state

        attention_mask = torch.cat([prompt_attention_mask, slot_attention_mask], 1)
        output_hidden_states = torch.cat([prompt_hidden_states, slot_hidden_states], 1)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=output_hidden_states), attention_mask

    def eval_rating(self, test_task_list=None, test_sample_numbers=None):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"rating": ["1-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")
        gt_ratings = []
        pred_ratings = []
        for i, (batch, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                prompt_ids = batch["input_ids"].to(self.model.device)
                slot_ids = batch["slot_ids"].to(self.model.device)
                encoder_outputs, attention_mask = self.encode(prompt_ids, slot_ids)
                results = self.model.generate_step(encoder_outputs=encoder_outputs,
                                                   attention_mask=attention_mask)
                gt_ratings.extend(batch['target_text'])
                pred_ratings.extend(results)

        predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
                            p in [str(i / 10.0) for i in list(range(10, 50))]]
        RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
        MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
        self.write_eval_result("sample_num: {}\nrating: ".format(len(dataloader)),
                               {"RMSE": 'RMSE {:7.4f}'.format(RMSE), "MAE": 'MAE {:7.4f}'.format(MAE)})

    def eval_sequential(self):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"sequential": ["2-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")
        all_info = []
        num_beams = 20
        for i, (batch, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                prompt_ids = batch["input_ids"].to(self.model.device)
                slot_ids = batch["slot_ids"].to(self.model.device)
                encoder_outputs, attention_mask = self.encode(prompt_ids, slot_ids)
                results = self.model.generate_step(encoder_outputs=encoder_outputs, attention_mask=attention_mask)
                beam_outputs = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    max_length=50,
                    num_beams=num_beams,
                    no_repeat_ngram_size=0,
                    num_return_sequences=num_beams,
                    early_stopping=True
                )
                generated_sents = self.model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                for j, item in enumerate(zip(results, batch['target_text'])):
                    new_info = {}
                    new_info['target_item'] = item[1]
                    new_info['gen_item_list'] = generated_sents[j * num_beams: (j + 1) * num_beams]
                    all_info.append(new_info)

        gt = {}
        ui_scores = {}
        for i, info in enumerate(all_info):
            gt[i] = [int(info['target_item'][len("item_"):])]
            pred_dict = {}
            for j in range(len(info['gen_item_list'])):
                try:
                    pred_dict[int(info['gen_item_list'][j][len("item_"):])] = -(j + 1)
                except:
                    pred_dict[0] = -(j + 1)
                    pass
            ui_scores[i] = pred_dict

        msg_5, res_5 = evaluate_all(ui_scores, gt, 5)
        msg_10, res_10 = evaluate_all(ui_scores, gt, 10)
        self.write_eval_result("sample_num:{}\nsequential@5: ".format(len(dataloader)), res_5)
        self.write_eval_result("sequential@10: ", res_10)

    def eval_explanation(self):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"explanation": ["3-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")

        tokens_predict = []
        tokens_test = []

        for i, (batch, _) in tqdm(enumerate(dataloader), desc="explanation"):
            with torch.no_grad():
                prompt_ids = batch["input_ids"].to(self.model.device)
                slot_ids = batch["slot_ids"].to(self.model.device)
                encoder_outputs, attention_mask = self.encode(prompt_ids, slot_ids)
                outputs = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    min_length=9,
                    num_beams=12,
                    num_return_sequences=1,
                    num_beam_groups=3,
                    repetition_penalty=0.7
                )
                results = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                tokens_predict.extend(results)
                tokens_test.extend(batch['target_text'])

        new_tokens_predict = [l.split() for l in tokens_predict]
        new_tokens_test = [ll.split() for ll in tokens_test]
        BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
        BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
        ROUGE = rouge_score(tokens_test, tokens_predict)

        result = {"BLEU-1": "{:7.4f}".format(BLEU1),
                  "BLEU-4": "{:7.4f}".format(BLEU4)}
        for (k, v) in ROUGE.items():
            result[k] = "{:7.4f}".format(v)
        self.write_eval_result("sample_num:{}\nexplanation".format(len(dataloader)), result)

    def eval_review(self):
        """Since T0 & GPT-2 checkpoints hosted on Hugging Face platform are slow to conduct inference,
        we only perform evaluation on the first 800 instances for prompts in Task Family 4."""
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"review": ["4-1"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")
        gt_ratings = []
        pred_ratings = []
        for i, (batch, _) in tqdm(enumerate(dataloader), desc="review"):
            if i > 50:
                break
            prompt_ids = batch["input_ids"].to(self.model.device)
            slot_ids = batch["slot_ids"].to(self.model.device)
            encoder_outputs, attention_mask = self.encode(prompt_ids, slot_ids)
            with torch.no_grad():
                results = self.model.generate_step(encoder_outputs=encoder_outputs,
                                                   attention_mask=attention_mask)
                gt_ratings.extend(batch['target_text'])
                pred_ratings.extend(results)

        pred_ratings = [p if p.isdigit() else 5 for p in pred_ratings]
        predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
        RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
        MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
        self.write_eval_result("sample_num:{}\nreview".format(len(dataloader)),
                               {"RMSE": "{:7.4f}".format(RMSE), "MAE": "{:7.4f}".format(MAE)})

    def eval_traditional(self):
        dataloader = get_loader(self.args, self.tokenizer, self.prompts, self.sample_numbers,
                                task_list={"traditional": ["5-2"]},
                                batch_size=self.args.test_batch_size,
                                mode="test")

        all_info = []
        num_beams = 20
        for i, (batch, _) in tqdm(enumerate(dataloader), desc="traditional"):
            with torch.no_grad():
                prompt_ids = batch["input_ids"].to(self.model.device)
                slot_ids = batch["slot_ids"].to(self.model.device)
                encoder_outputs, attention_mask = self.encode(prompt_ids, slot_ids)

                results = self.model.generate_step(encoder_outputs=encoder_outputs,
                                                   attention_mask=attention_mask)
                beam_outputs = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    max_length=50,
                    num_beams=num_beams,
                    no_repeat_ngram_size=0,
                    num_return_sequences=num_beams,
                    early_stopping=True
                )
                generated_sents = self.model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                for j, item in enumerate(zip(results, batch['target_text'])):
                    new_info = {}
                    new_info['target_item'] = item[1]
                    new_info['gen_item_list'] = generated_sents[j * num_beams: (j + 1) * num_beams]
                    all_info.append(new_info)

        gt = {}
        ui_scores = {}
        for i, info in enumerate(all_info):
            gt[i] = [int(info['target_item'][len("item_"):])]
            pred_dict = {}
            for j in range(len(info['gen_item_list'])):
                try:
                    pred_dict[int(info['gen_item_list'][j][len("item_"):])] = -(j + 1)
                except:
                    pred_dict[0] = -(j + 1)
            ui_scores[i] = pred_dict

        msg_1, res_1 = evaluate_all(ui_scores, gt, 1)
        msg_5, res_5 = evaluate_all(ui_scores, gt, 5)
        msg_10, res_10 = evaluate_all(ui_scores, gt, 10)
        self.write_eval_result("sample_num:{}\ntraditional@1: ".format(len(dataloader)), res_1)
        self.write_eval_result("traditional@5: ", res_5)
        self.write_eval_result("traditional@10: ", res_10)


def main():
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S",
                        level=logging.INFO)
    # args supplementary
    args = parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.world_size = args.n_gpu
    args.local_rank = int(os.environ.get("RANK", "0"))
    if args.local_rank in [-1, 0]:
        logging.info(f"{args}\n")

    # seet seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # change working directory to the project root
    project_dir = Path(__file__).resolve().parent.parent
    os.chdir(project_dir)
    if args.local_rank in [-1, 0]:
        logging.info("current working directory: {}\n".format(project_dir))

    # dist initialization
    if args.distributed:
        logging.info(f"initializing gpu {args.local_rank}...")
        torch.cuda.set_device(args.local_rank)
        backend = "gloo" if args.n_gpu <= 0 else "nccl"
        dist.init_process_group(backend=backend, rank=args.local_rank, world_size=args.world_size)

    args.cache_dir = os.path.join("cached", "t5-base2")
    args.ckpt_path = os.path.join(args.project_root, "cached", args.ckpt)
    args.cache_dir = os.path.join(args.project_root, "cached", args.backbone)
    args.data_dir = os.path.join(args.project_root, "data", args.data_name)
    args.eval_dir = os.path.join(args.project_root, args.eval_dir, args.data_name)
    args.loaded_model = args.ckpt_path.split("/")[-1][:-len(".pth")]

    # create model & load model
    if args.model_class == "control_rec":
        model_class = ControlRecPretraining
    elif args.model_class == "p5":
        model_class = P5Pretraining
    else:
        raise NotImplementedError

    pretrained_model = PretrainedModel(args, model_class=model_class)
    if args.n_gpu > 0:
        pretrained_model.model.to(args.local_rank)

    # evaluator
    dump_name = f"{args.model_class}-{args.ckpt.split('/')[-1]}"
    dump_name = dump_name[:-len(".pth")]
    if args.model_class == "control_rec":
        evaluator = CPLRecSysEvaluator(args, pretrained_model.model, dump_name)
    elif args.model_class == "p5":
        evaluator = P5Evaluator(args, pretrained_model.model, dump_name)
    else:
        raise ValueError

    """test rating"""
    evaluator.eval_rating()

    """test sequential"""
    evaluator.eval_sequential()

    """test explanation"""
    evaluator.eval_explanation()

    """test review"""
    # Since T0 & GPT-2 checkpoints hosted on Hugging Face platform are slow to conduct inference,
    # we only perform evaluation on the first 800 instances for prompts in Task Family 4.
    evaluator.eval_review()

    """test traditional"""
    evaluator.eval_traditional()


if __name__ == '__main__':
    main()
