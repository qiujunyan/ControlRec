import json
import logging
import os
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
import random
import numpy as np

import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

project_dir = os.path.join(os.path.dirname(__file__), "../")
os.chdir(project_dir)
sys.path.append(project_dir)

from param import parse_args
from pretrain_data import get_loader
from utils import TrainLogger, load_state_dict, LossMeter
from dist_utils import reduce_dict
from models import ControlRecPretraining, P5Pretraining

from tokenization import P5Tokenizer, T5Tokenizer

# from dist_utils import reduce_dict
# from param import parse_args
# from pretrain_data import get_loader
# from utils import LossMeter, TrainLogger

warnings.filterwarnings("ignore", category=UserWarning)

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.local_rank = args.local_rank
        self.verbose = True
        # if not self.verbose:
        #     set_global_logging_level(logging.ERROR, ["transformers"])

    def create_config(self):
        from transformers import T5Config
        if 't5' in self.args.backbone:
            config_class = T5Config
        else:
            return None

        config = config_class.from_pretrained(self.args.backbone,
                                              cache_dir=self.args.cache_dir,
                                              local_files_only=self.args.local_files_only)

        args = self.args

        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout
        config.losses = args.task
        config.eps = args.eps_for_ce

        return config

    def create_model(self, model_class, config=None, **kwargs):
        # print(f'Building Model at GPU {self.args.local_rank}')
        model_name = self.args.backbone

        # cache_dir = os.path.join("cached", self.args.backbone)
        if not os.path.isdir(self.args.cache_dir):
            os.makedirs(self.args.cache_dir, exist_ok=True)

        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.args.cache_dir,
            config=config,
            local_files_only=self.args.local_files_only,
            **kwargs
        )
        # freeze T5 parameters except embedding
        for name, param in model.named_parameters():
            if name != "shared.weight" and self.args.freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if self.args.local_rank == 0:
            logging.info(f"number of parameters: {model.num_parameters()}")
        return model

    def create_optimizer_and_scheduler(self):
        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import get_linear_schedule_with_warmup
            # from transformers.optimization import AdamW
            from torch.optim import AdamW

            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)

            self.logging("Batch per epoch: %d" % batch_per_epoch)
            self.logging("Total Iters: %d" % t_total)
            self.logging(f'Warmup ratio: {warmup_ratio}')
            self.logging("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_linear_schedule_with_warmup(
                optim, warmup_iters, t_total)

        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        results = self.model.load_state_dict(state_dict, strict=False)

        self.logging(f'Model loaded from {ckpt_path}')
        self.logging(results)

    def init_weights(self):
        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.model.apply(init_bert_weights)
        self.model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.dump_dir):
            os.makedirs(self.args.dump_dir, exist_ok=True)
        class_name = self.model.name if not self.args.distributed else self.model.module.name
        dump_name = f"{class_name}-{name}"
        dump_file = os.path.join(self.args.dump_dir, f"{dump_name}.pth")
        torch.save(self.model.state_dict(), dump_file)
        self.logging(f"Model {dump_name} dumped: {dump_file}")

    def logging(self, msg):
        if self.local_rank == 0:
            logging.info(f"local_rank {self.local_rank}: {msg}")

    def debug_info(self, msg):
        logging.info(f"local_rank {self.local_rank}: {msg}")


class Trainer(TrainerBase):
    def __init__(self, args, model_class, tokenizer, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader)
        self.tasks = self.args.task.split(",")
        self.losses = ["gen", "pcl", "slot", "desc"] + self.tasks
        model_kwargs = {}
        config = self.create_config()
        self.model = self.create_model(model_class, config, **model_kwargs)
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.tokenizer = tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.ckpt is not None:
            ckpt_path = os.path.join(args.load_dir, args.ckpt)
            self.load_checkpoint(ckpt_path)
            # if "Epoch-" in args.load_dir:
            #     self.start_epoch = int(args.load.split('Epoch-')[-1])

        if self.args.from_scratch:
            self.init_weights()

        if args.distributed and args.n_gpu > 0:
            self.model = self.model.to(args.local_rank)

        if args.multiGPU and args.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

        # tensorboard
        self.logger = TrainLogger(log_dir=os.path.join(self.args.project_root, args.log_dir),
                                  dataset=args.train,
                                  local_rank=args.local_rank,
                                  verbose=args.verbose)

    def train(self):
        # loss_meters = {task: LossMeter() for task in self.tasks}
        loss_meters = defaultdict(LossMeter)
        best_eval_loss = 100000.

        global_step = 0
        total_steps = len(self.train_loader) * self.args.epoch
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self.model.train()

            epoch_results = {}
            for task in self.losses:
                epoch_results[task] = 0.
                epoch_results[f'{task}_count'] = 0

            self.logging(f"total batch: {len(self.train_loader)}")

            for step_i, batch in enumerate(self.train_loader):
                if self.args.distributed:
                    results = self.model.module.train_step(batch, self.args.do_IDM, self.args.do_ICL,
                                                           tau=self.args.tau, alpha=step_i / total_steps)
                    if isinstance(results, tuple):
                        results, output = results
                else:
                    results = self.model.train_step(batch,
                                                    self.args.do_IDM,
                                                    self.args.do_ICL,
                                                    tau=self.args.tau,
                                                    alpha=step_i / total_steps)

                # gradually increasing the weight of pcl
                loss = results['loss']
                loss.backward()

                # # check gradient
                # for name, params in self.model.named_parameters():
                #     if params.grad is not None and torch.any(torch.isnan(params.grad)):
                #         target_pooled_hidden_states = output.target_pooled_hidden_states
                #         candidate_prompt_labels = output.candidate_prompt_labels
                #         pcl_pooled_hidden_states = output.pcl_pooled_hidden_states
                #         pcl_similarity = torch.matmul(target_pooled_hidden_states,
                #                                       pcl_pooled_hidden_states.transpose(1, 2))
                #         pcl_similarity = pcl_similarity.squeeze(1) / self.args.tau
                #         from utils import cross_entropy_loss
                #         pcl_loss = cross_entropy_loss(candidate_prompt_labels, pcl_similarity,
                #                                       reduction="sum", eps=self.config.eps)
                #         self.debug_info(f"gradient exception observed...")
                #         self.save(f"Nan-Rank{self.local_rank}-Epoch0{epoch + 1}-Step{step_i}")
                #         self.save_break_point_batch(batch, step_i)
                #         for param in self.model.parameters():
                #             param.grad = None
                #         self.monitor_grad(global_step)
                #         self.monitor_param(global_step)

                if step_i != 0 and self.args.save_every_n_step > 0 and step_i % self.args.save_every_n_step == 0:
                    self.save(f"Epoch0{epoch + 1}-Step{step_i}")
                    self.monitor_grad(global_step=global_step)
                    self.monitor_param(global_step=global_step)

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                self.optim.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if step_i % self.args.print_every_n_step == 0:
                    desc_str = f'Epoch: {epoch}-Step: {step_i}-LR: {lr:.6f}'
                    total_loss = 0
                    for loss_name in self.losses:
                        if loss_name in results.keys():
                            loss_meters[loss_name].update(results[f"{loss_name}"] / results[f"{loss_name}_count"])
                        if loss_name not in self.tasks and loss_meters[loss_name].val > 0:
                            desc_str += f"\t{loss_name}({epoch_results[loss_name + '_count']}): {loss_meters[loss_name].val:.3f} "
                            total_loss += loss_meters[loss_name].val
                        if len(loss_meters[loss_name]) > 0:
                            self.logger.add_scalars(main_tag="train",
                                                    tag_scalar_dict={loss_name: loss_meters[f"{loss_name}"].val},
                                                    global_step=epoch_results[f"{loss_name}_count"])
                    desc_str += f"\tloss: {total_loss:.3f}"
                    self.logger.add_scalars(main_tag="train",
                                            tag_scalar_dict={"loss": total_loss},
                                            global_step=global_step)
                    # if self.local_rank == 0:
                    self.logging(desc_str)
                    # logging.info(f"local_rank: {self.local_rank}, {desc_str}")

                if self.args.distributed:
                    dist.barrier()

            # validation
            if self.args.distributed:
                dist.barrier()
            self.save("Epoch%02d" % (epoch + 1))
            valid_results = self.evaluate_epoch(epoch=epoch)
            valid_results = reduce_dict(valid_results, average=False)

            valid_loss = valid_results['total_loss']
            valid_loss_count = valid_results['total_loss_count']

            avg_valid_loss = valid_loss / valid_loss_count
            losses_str = f"Epoch {epoch} Valid Loss: {avg_valid_loss:.3f}\n"

            for name, loss in valid_results.items():
                if name[-5:] != 'count':
                    loss_count = int(valid_results[name + '_count'])
                    if loss_count > 0:
                        avg_loss = loss / loss_count
                        losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

            losses_str += '\n'
            self.logging(losses_str)

            if avg_valid_loss < best_eval_loss:
                best_eval_loss = avg_valid_loss
                self.save("BEST_EVAL_LOSS")

            if self.args.distributed:
                dist.barrier()

    def evaluate_epoch(self, epoch):
        epoch_results = {"total_loss": 0, "total_loss_count": 0}
        for loss_name in self.tasks:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.logging("start validation")
        self.logging(f"total valid iters: {len(self.val_loader)}")
        self.model.eval()
        with torch.no_grad():
            # loss_meter = LossMeter()
            # loss_meters = [LossMeter() for _ in range(len(self.tasks))]
            # pbar = tqdm(total=len(self.val_loader), ncols=275)
            for step_i, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                if self.args.valid_ratio < random.random():
                    continue

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

            if self.args.distributed:
                dist.barrier()
            return epoch_results

    def monitor_grad(self, global_step):
        for name, params in self.model.named_parameters():
            if params.grad is not None and not torch.any(torch.isnan(params.grad)):
                self.logger.add_histgram("grad", values=params.grad, global_step=global_step)

    def monitor_param(self, global_step):
        for name, params in self.model.named_parameters():
            if params is not None and not torch.any(torch.isnan(params)):
                self.logger.add_histgram("data", values=params, global_step=global_step)

    def save_break_point_batch(self, batch, step_i):
        if not os.path.isdir(self.args.dump_dir):
            os.makedirs(self.args.dump_dir, exist_ok=True)
        dump_file = os.path.join(self.args.dump_dir, f"rank{self.local_rank}-batch{step_i}.pkl")
        with open(dump_file, "wb") as fp:
            pickle.dump(batch, fp)
        self.debug_info(f"batch {dump_file} saved")


def main_worker(local_rank, args, template):
    if args.distributed:
        logging.info(f"initializing gpu {local_rank}...")
        torch.cuda.set_device(local_rank)
        backend = "gloo" if args.n_gpu <= 0 else "nccl"
        dist.init_process_group(backend=backend, rank=local_rank, world_size=args.world_size)

    if 't5' in args.backbone:
        # additional_special_tokens = ["[name]", "[city]", "[state]", "[postal_code]", "[categories]"]
        # additional_special_tokens = ["extra_id_NAME"]
        # import pdb
        # pdb.set_trace()
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone,
            model_max_length=args.max_input_length,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir,
            truncation=True,
            local_files_only=args.local_files_only)
    else:
        raise NotImplementedError()

    if args.model_class != "control_rec":
        args.do_ICL = False
        args.do_IDM = False

    # if greater than 1, a data sample will be used for multiple times with different prompts in certain task family
    train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 5, 5), 'explanation': 1, 'review': 1,
                            'traditional': (5, 5, 5), 'item_desc': 5, "seq_desc": 5}
    if args.data_name == "yelp":
        task_list = {'rating': ['1-1', '1-2', '1-3'],
                     'sequential': ['2-1', '2-2', '2-3', '2-4'],
                     'explanation': ['3-1', '3-2', '3-3', '3-4'],
                     'review': ['4-1', '4-2'],
                     'traditional': ['5-1', '5-2', '5-3'],
                     'item_desc': ['6-1'],
                     'seq_desc': ['7-1']
                     }
    elif args.data_name in ["beauty", "toys", "sports"]:
        task_list = {'rating': ['1-1', '1-2', '1-3'],
                     'sequential': ['2-1', '2-2', '2-3', '2-4'],
                     'explanation': ['3-1', '3-2', '3-3', '3-4'],
                     'review': ['4-1', '4-2'],
                     'traditional': ['5-1', '5-2', '5-3'],
                     'item_desc': ['6-1'],
                     'seq_desc': ['7-1']
                     }
    else:
        raise NotImplementedError

    train_loader = get_loader(
        args,
        tokenizer,
        template,
        train_sample_numbers,
        task_list=task_list,
        batch_size=args.train_batch_size,
        mode='train'
    )

    val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1, 1), 'explanation': 1, 'review': 1,
                          'traditional': (1, 1, 1), "item_desc": 1, "seq_desc": 1}
    val_loader = get_loader(
        args,
        tokenizer,
        template,
        val_sample_numbers,
        task_list=task_list,
        batch_size=args.valid_batch_size,
        mode='val'
    )

    if args.model_class == "control_rec":
        model_class = ControlRecPretraining
    elif args.model_class == "p5":
        model_class = P5Pretraining
    else:
        raise NotImplementedError
    trainer = Trainer(args,
                      model_class=model_class,
                      tokenizer=tokenizer,
                      train_loader=train_loader,
                      val_loader=val_loader)
    trainer.train()


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

    args.dump_dir = os.path.join(args.project_root, args.dump_dir, args.data_name)
    args.cache_dir = os.path.join(args.project_root, "cached", args.backbone)
    args.load_dir = os.path.join(args.project_root, "cached")

    if args.trigger_as_prompt:
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
                                             in enumerate(group["completion"])]
                template[task_name] = task_prompt
    main_worker(args.local_rank, args, template)


if __name__ == '__main__':
    main()
