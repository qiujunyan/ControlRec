import re
import numpy as np
import torch
import torch.distributed as dist
import collections
import logging
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import gzip
import pickle
import json
import torch.nn.functional as F


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def str2bool(str_):
    if str_.lower() == "false":
        return False
    elif str_.lower() == "true":
        return True


def merge_dict(dict_list):
    res = {}
    for dict_ in dict_list:
        for key, value in dict_.items():
            res[key] = value
    return res


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals) if len(self.vals) != 0 else 0

    def __repr__(self):
        return str(self.val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def cross_entropy_loss(label, logits, reduction="mean", eps=0):
    prob = torch.softmax(logits, -1)
    truncate_prob = torch.clip(prob, min=eps, max=1)
    log_logits = torch.log(truncate_prob)
    loss = -label.to(torch.float32) * log_logits
    if reduction == "none" or reduction is None:
        loss = loss.sum(-1)
    elif reduction == "mean":
        loss = loss.sum(-1) / len(loss)
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def cross_entropy_loss_with_mask(label, logits, eps=0):
    prob = torch.softmax(logits, -1)
    truncate_prob = torch.clip(prob, min=eps, max=1)
    log_logits = torch.log(truncate_prob)

    label_without_neg = torch.max(label, torch.zeros_like(label))
    onehot_label = F.one_hot(label_without_neg, num_classes=logits.shape[-1])

    # loss.shape -> (batch_size, seq_len, vocab_size)
    loss = -onehot_label.to(torch.float32) * log_logits
    ignore_index = -100
    mask = torch.eq(label, ignore_index)
    loss = loss.masked_fill(mask.unsqueeze(-1), 0)
    return loss.sum(-1)


class TrainLogger(object):
    def __init__(self, log_dir, dataset, local_rank, verbose=True):
        super().__init__()
        self.local_rank = local_rank
        self.verbose = verbose
        self.log_dir = self.create_log_dir(log_dir, dataset)
        self.writer = SummaryWriter(self.log_dir) if self.local_rank == 0 and self.verbose else None

    def create_log_dir(self, log_dir, dataset):
        cur_log_dir = None
        if self.local_rank == 0 and self.verbose:
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if dataset != "yelp":
                dataset = "amazon-" + dataset
            cur_log_dir = os.path.join(log_dir, dataset, cur_time)
            if not os.path.isdir(cur_log_dir):
                os.makedirs(cur_log_dir, exist_ok=True)
            logging.info(f"logging files {cur_log_dir}")
        return cur_log_dir

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        if self.local_rank == 0 and self.verbose:
            self.writer.add_scalars(main_tag=main_tag,
                                    tag_scalar_dict=tag_scalar_dict,
                                    global_step=global_step)

    def add_scalar(self, tag, scalar_value, global_step):
        if self.local_rank == 0 and self.verbose:
            self.writer.add_scalar(tag=tag,
                                   scalar_value=scalar_value,
                                   global_step=global_step)

    def add_histgram(self, tag, values, global_step):
        if self.local_rank == 0 and self.verbose:
            self.writer.add_histogram(tag=tag,
                                      values=values,
                                      global_step=global_step)
