import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='yelp')
    parser.add_argument("--valid", default='yelp')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--data_name', default="beauty", choices=["yelp", "toys", "sports", "beauty"])
    parser.add_argument('--model_class', default="control_rec", choices=["control_rec", "p5"])

    # Paths
    parser.add_argument("--project_root", type=str, default="./")
    # parser.add_argument("--data_dir", type=str, default="cached")
    parser.add_argument("--dump_dir", type=str, default="cached")
    parser.add_argument("--eval_dir", type=str, default="evaluate")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument("--local_files_only", type=str2bool, default=True)

    # CPU/GPU
    parser.add_argument("--multiGPU", type=str2bool, default=True)
    parser.add_argument('--fp16', type=str2bool, default=False)
    parser.add_argument("--distributed", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument('--local_rank', type=int)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    # parser.add_argument('--tokenizer', type=str, default='p5')

    # Training
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--task", default="rating,sequential,explanation,review,traditional", type=str)
    parser.add_argument('--log_train_accuracy', action='store_true')
    parser.add_argument("--valid_ratio", type=float, default=0.05)

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_gen_length', type=int, default=16)
    parser.add_argument('--max_input_length', type=int, default=512)

    # Data
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument("--rating_augment", action="store_true")

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')
    parser.add_argument("--trigger_as_prompt", type=str2bool, default=True)
    parser.add_argument("--all_data_file", type=str, default="all_datas.pkl")
    parser.add_argument("--freeze", type=str2bool, default=False)
    parser.add_argument("--eps_for_ce", type=float, default=0)
    parser.add_argument("--verbose", type=str2bool, default=False)

    # Logging
    parser.add_argument("--log_dir", type=str, default="run")
    parser.add_argument("--print_every_n_step", type=int, default=1)
    parser.add_argument("--eval_every_n_step", type=int, default=1000)
    parser.add_argument("--save_every_n_step", type=int, default=1000)

    # Ablation
    parser.add_argument('--neg_samples_for_hfm', type=int, default=10)
    parser.add_argument('--neg_samples_for_icl', type=int, default=5)
    parser.add_argument("--do_IDM", type=str2bool, default=True)
    parser.add_argument("--do_ICL", type=str2bool, default=True)

    # Hope params
    parser.add_argument('--dist_url', type=str)
    parser.add_argument('--gpu_count', type=int, default=1)

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
