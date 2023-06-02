import gzip
import json
import logging
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


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


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, prompts, sample_numbers, task_list,
                 split="yelp", mode="train", sample_type="random"):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.sample_numbers = sample_numbers
        self.task_list = task_list
        self.prompts_by_group_id = {}
        for prompts in self.prompts.values():
            for group_id, group in prompts.items():
                self.prompts_by_group_id[group_id] = group
        self.split = split
        self.mode = mode
        self.rating_augment = args.rating_augment
        self.sample_type = sample_type
        self.neg_samples_for_hfm = self.args.neg_samples_for_hfm
        self.neg_samples_for_icl = self.args.neg_samples_for_icl
        self.do_icl = self.args.do_ICL and self.mode == "train"
        self.do_hfm = self.args.do_IDM and self.mode == "train"

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token = self.tokenizer.additional_special_tokens[0]
        self.sep_token = self.tokenizer.additional_special_tokens[0]  # used to separate user and item ids
        self.user_token = "user"
        self.item_token = "item"
        self.user_subtoken = self.tokenizer.tokenize(self.user_token)[0]

        # prompt groups for contrastive prompt learning
        # structure: {input_format: {output_format: [prompt_group_ids]}}
        self.prompt_groups = {"user+item": {
            "star": ["1-1"],
            "yes/no": ["1-2", "1-3", "5-1"],
            "explanation": ["3-1", "3-2", "3-3", "3-4"]
        },
            "user+purchase": {
                "item": ["2-1"],
                "yes/no": ["2-3"]
            },
            "user+purchase+candidate": {
                "item": ["2-2"],
                "yes/no": ["2-4"]
            },
            "user": {
                "star": ["4-1"],
                "yes/no": ["4-2"]
            },
            "user+candidate": {
                "item": ["5-2"],
                "yes/no": ["5-3"]
            }
        }

    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = -1
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁') or tokenized_text[i] in [self.sep_token, self.cls_token]:
                curr += 1
            whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1]

    def calculate_visible_matrix(self, tokenized_text, whole_word_ids):
        # if cur subword is from user or cls or sep token, set to 1, else 0
        assert len(tokenized_text) == len(whole_word_ids)
        user_matrix = []
        user_word_id = -1
        for subword, whole_word_id in zip(tokenized_text, whole_word_ids):
            if subword in [self.sep_token, self.cls_token]:
                user_matrix.append(1)
            elif subword == self.user_subtoken:
                user_word_id = whole_word_id
                user_matrix.append(1)
            elif user_word_id == whole_word_id:
                user_matrix.append(1)
            else:
                user_matrix.append(0)

        seq_len = len(tokenized_text)
        visible_matrix = torch.ones(seq_len, seq_len, dtype=torch.int64)
        for i, idx in enumerate(user_matrix):
            if idx != 0:
                continue
            for j in range(seq_len):
                if user_matrix[j] == 0 and whole_word_ids[j] != whole_word_ids[i]:
                    visible_matrix[i][j] = 0
        return visible_matrix


class YelpDataset(BaseDataset):
    def __init__(self, args, tokenizer, prompts, sample_numbers, task_list,
                 split="yelp", mode="train", sample_type="random"):
        super().__init__(args, tokenizer, prompts, sample_numbers, task_list,
                         split=split, mode=mode, sample_type=sample_type)
        self.all_datas = self.load_data()
        self.all_item, self.probability, self.user_items = self.get_user_items()
        if mode == "test":
            self.negative_samples = ReadLineFromFile(os.path.join(self.args.project_root, 'data',
                                                                  split, 'negative_samples.txt'))
        # metad data
        self.user2id, self.item2id, self.user_list, self.item_list, self.id2item, \
            self.user_id2name, self.meta_data, self.user_data, self.meta_dict, self.user_meta_dict = self.load_meta_data()
        self.num_users, self.num_items = len(self.user2id), len(self.item2id)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token = self.tokenizer.additional_special_tokens[0]
        self.sep_token = self.tokenizer.additional_special_tokens[1]  # used to separate user and item ids
        self.user_token = "user"
        self.item_token = "item"

        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        self.feature_field = ["name", "address", "city", "state", "postal_code", "categories"]

        self.datas, self.task_dict = self.try_cache()
        assert len(self.datas) == self.total_length
        if self.args.local_rank == 0:
            logging.info("data source: {}".format(split.split(",")))
            logging.info(f"total {mode} number: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        batch_data = self.datas[idx]
        task_name = batch_data["task"]
        task_id = batch_data["task_id"]
        input_text = batch_data["input_text"]
        slot_text = batch_data["slot_text"]
        target_text = batch_data["target_text"]

        assert input_text is not None and slot_text is not None
        input_text = f"{self.cls_token} " + input_text
        slot_text = f"{self.cls_token} " + slot_text
        input_ids = self.tokenizer.encode(input_text, padding=True, truncation=True,
                                          max_length=self.args.max_input_length)
        slot_ids = self.tokenizer.encode(slot_text, padding=True, truncation=True,
                                         max_length=self.args.max_input_length, )
        if target_text is not None:
            target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True,
                                               max_length=self.args.max_gen_length)
        else:
            target_ids = None

        candidate_desc_samples = batch_data["candidate_desc_samples"]
        candidate_slot_samples = batch_data["candidate_slot_samples"]

        pcl_labels, prompt_ids_list, prompt_pool, prompt_lens = None, None, None, None
        if task_name in ["item_desc", "seq_desc"]:
            candidate_desc_ids = [self.tokenizer.encode(sample, max_length=self.args.max_input_length,
                                                        truncation=True) for sample in
                                  candidate_desc_samples]
            candidate_desc_ids = [torch.LongTensor(desc_id) for desc_id in candidate_desc_ids]
            candidate_desc_lens = [len(desc) for desc in candidate_desc_ids]

            candidate_slot_ids = [self.tokenizer.encode(sample, max_length=self.args.max_input_length,
                                                        truncation=True) for sample in
                                  candidate_slot_samples]
            candidate_slot_ids = [torch.LongTensor(item_id) for item_id in candidate_slot_ids]
            candidate_slot_lens = [len(item) for item in candidate_slot_ids]

        else:
            candidate_desc_ids, candidate_slot_ids = None, None
            candidate_desc_lens, candidate_slot_lens = 0, 0

            if self.do_icl:
                # datas for contrastive instruction learning
                neg_pool, pos_group_id = self.construct_prompt_pool(task_id)
                neg_sample_ids = []
                for neg_group_id, num in neg_pool.items():
                    if neg_group_id not in self.task_dict.keys():
                        continue
                    neg_sample_ids.extend(random.sample(self.task_dict[neg_group_id], num))

                prompt_pool = []
                for neg_id in neg_sample_ids:
                    prompt_pool.append(self.datas[neg_id]["input_text"])

                pos_prompt_id = random.sample(self.task_dict[pos_group_id], 1)[0]
                while pos_prompt_id == idx:
                    pos_prompt_id = random.sample(self.task_dict[pos_group_id], 1)[0]
                pos_prompt = self.datas[pos_prompt_id]["input_text"]
                prompt_pool.append(pos_prompt)
                random.shuffle(prompt_pool)
                prompt_ids_list = [torch.LongTensor(self.tokenizer.encode(t, max_length=self.args.max_input_length,
                                                                          truncation=True)) for t in prompt_pool]
                prompt_lens = [len(prompt) for prompt in prompt_ids_list]
                pcl_labels = [int(prompt == pos_prompt) for prompt in prompt_pool]
                assert sum(pcl_labels) == 1

        neg_desc_labels = batch_data["neg_desc_labels"]
        neg_slot_labels = batch_data["neg_slot_labels"]

        out_dict = {'input_ids': torch.LongTensor(input_ids),
                    'input_length': len(input_ids),
                    'target_ids': torch.LongTensor(target_ids) if target_ids is not None else None,
                    'target_length': len(target_ids) if target_ids is not None else 0,
                    "slot_ids": torch.LongTensor(slot_ids),
                    "slot_length": len(slot_ids),
                    "candidate_slot_ids": candidate_slot_ids,
                    "candidate_slot_length": candidate_slot_lens,
                    "candidate_desc_ids": candidate_desc_ids,
                    "candidate_desc_length": candidate_desc_lens,
                    'target_text': target_text,
                    "input_text": input_text,
                    'slot_text': slot_text,
                    "candidate_slot_text": candidate_slot_samples,
                    "candidate_desc_test": candidate_desc_samples,
                    "neg_desc_labels": neg_desc_labels,
                    "neg_item_labels": neg_slot_labels,
                    "pcl_labels": pcl_labels,
                    "prompt_ids": prompt_ids_list,
                    "prompt_text": prompt_pool,
                    "prompt_lens": prompt_lens,
                    'task': task_name}
        return out_dict

    def __construct_item_desc(self, item_datum):
        # features of items
        feature_template = "[name], located in the city of [city], [state]. Its postal code is [postal_code] and " \
                           "it falls under the category of [categories]"
        for feature in self.feature_field:
            if feature in item_datum:
                value = item_datum[feature]
                if value is not None:
                    feature_template = feature_template.replace("[{}]".format(feature), value)
        return feature_template

    def load_data(self):
        all_datas = {}
        all_datas["review"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                       self.split, 'review_splits.pkl'))[self.mode]
        all_datas["explanation"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                            self.split, 'exp_splits.pkl'))[self.mode]
        if self.rating_augment:
            all_datas["rating"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                           self.split, 'rating_splits_augmented.pkl'))[self.mode]
        else:
            all_datas["rating"] = all_datas["review"]
        all_datas["sequential"] = ReadLineFromFile(os.path.join(self.args.project_root, 'data',
                                                                self.split, 'sequential_data.txt'))
        return all_datas

    def try_cache(self):
        all_datas_path = os.path.join(self.args.project_root, "data", self.split,
                                      f"{self.mode}_" + self.args.all_data_file)
        if self.mode == "test" or not os.path.exists(all_datas_path):
            if self.args.local_rank == 0:
                logging.info(f"processing {self.mode} datas")
            all_datas = []
            task_dict = defaultdict(list)
            for idx, datum_info_idx in tqdm(enumerate(self.datum_info), total=len(self.datum_info)):
                data_entry = {}
                assert datum_info_idx[0] == idx
                if len(datum_info_idx) == 3:
                    task_name = datum_info_idx[1]
                    datum_idx = datum_info_idx[2]
                elif len(datum_info_idx) == 4:
                    task_name = datum_info_idx[1]
                    datum_idx = datum_info_idx[2]
                    task_id = datum_info_idx[3]
                else:
                    raise NotImplementedError
                data_entry["task"] = task_name
                data_entry["idx"] = idx
                data_entry["datum_idx"] = datum_info_idx[2]

                slot_text, target_text, input_text = None, None, None
                neg_desc_labels, neg_slot_labels = None, None
                candidate_desc_samples, candidate_slot_samples = None, None
                task_id = None

                if task_name == 'rating':
                    rating_datum = self.all_datas["rating"][datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]
                    group = self.prompts["rating"][task_id]

                    user_id = f"{self.user_token}_" + self.user2id[rating_datum["reviewerID"]]
                    target_id = f"{self.item_token}_" + self.item2id[rating_datum["asin"]]
                    prompt_id = random.randint(0, len(group) - 1)
                    if task_id == "1-1":
                        slot_text = f" {self.sep_token} ".join([user_id, target_id])
                        target_text = self.gaussian_sampling(rating_datum)
                        target_text = str(target_text)
                        input_text = group[prompt_id]

                    elif task_id == "1-2":
                        slot_text = f" {self.sep_token} ".join([user_id, target_id])
                        rand_prob = random.random()
                        if rand_prob > 0.5:
                            target_text = "yes"
                            input_text = group[prompt_id].format(int(rating_datum["overall"]))
                        else:
                            target_text = "no"
                            overall_candidates = [_ for _ in range(0 + 1, 5 + 1) if _ != int(rating_datum['overall'])]
                            overall_idx = random.randint(0, len(overall_candidates) - 1)
                            input_text = group[prompt_id].format(overall_candidates[overall_idx])

                    elif task_id == "1-3":
                        slot_text = f" {self.sep_token} ".join([user_id, target_id])
                        if rating_datum["overall"] >= 4.0:
                            target_text = "yes"
                        else:
                            target_text = "no"
                        input_text = group[prompt_id]

                    else:
                        raise NotImplementedError("rating")

                elif task_name == "sequential":
                    sequential_datum = self.all_datas["sequential"][datum_idx]
                    sequence = sequential_datum.split()
                    user_id = sequence[0]
                    user_desc = self.user_id2name[user_id]
                    history_limit = 20

                    if self.mode == "train":
                        end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                        end_index = random.randint(0, len(end_candidates) - 1)
                        end_pos = end_candidates[end_index]
                        start_candidates = [_ for _ in range(1, min(4, end_pos))]
                        start_index = random.randint(0, len(start_candidates) - 1)
                        start_pos = start_candidates[start_index]
                        purchase_history = sequence[start_pos:end_pos + 1]
                        target_item = sequence[end_pos + 1]
                    elif self.mode == 'val':
                        purchase_history = sequence[1:-2]
                        target_item = sequence[-2]
                    elif self.mode == 'test':
                        purchase_history = sequence[1:-1]
                        target_item = sequence[-1]
                    else:
                        raise NotImplementedError

                    if len(purchase_history) > history_limit:
                        purchase_history = purchase_history[-history_limit:]
                    target_item = f"{self.item_token}_" + target_item
                    purchase_history = [f"{self.item_token}_" + item for item in purchase_history]

                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)
                    if task_id == "2-1":
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history)])
                        target_text = target_item
                        input_text = group[prompt_id]

                    elif task_id == "2-2":
                        if self.mode in ['train', 'val']:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = random.randint(79, 99)
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                        elif self.mode == 'test':
                            assert user_id == self.negative_samples[int(user_id) - 1].split(' ', 1)[0]
                            candidate_samples = self.negative_samples[int(user_id) - 1].split(' ', 1)[1].split(' ')
                        else:
                            raise NotImplementedError
                        candidate_samples.extend([target_item])
                        random.shuffle(candidate_samples)
                        rand_prob = random.random()

                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history), " ".join(candidate_samples)])
                        target_text = target_item
                        input_text = group[prompt_id]

                    elif task_id == "2-3":
                        rand_prob = random.random()
                        if rand_prob > 0.5:
                            slot_text = f" {self.sep_token} ".join(
                                [f"{self.user_token}_" + user_id, " ".join(purchase_history),
                                 target_item])
                            target_text = "yes"
                        else:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = 1
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                            candidate_samples = [f"{self.user_token}_" + sample for sample in candidate_samples]
                            slot_text = f" {self.sep_token} ".join(
                                [f"{self.user_token}_" + user_id, " ".join(purchase_history),
                                 candidate_samples[0]])
                            target_text = "no"
                        input_text = group[prompt_id]

                    elif task_id == "2-4":
                        if self.mode in ['train', 'val']:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = random.randint(79, 99)
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                        elif self.mode == 'test':
                            assert user_id == self.negative_samples[int(user_id) - 1].split(' ', 1)[0]
                            candidate_samples = self.negative_samples[int(user_id) - 1].split(' ', 1)[1].split(' ')
                        else:
                            raise NotImplementedError

                        if random.random() > 0.5:
                            candidate_samples.extend([target_item])
                            random.shuffle(candidate_samples)
                            target_text = "yes"
                        else:
                            target_text = "no"

                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history), " ".join(candidate_samples)])
                        input_text = group[prompt_id]

                    else:
                        raise NotImplementedError("sequential recommendation")

                elif task_name == "explanation":
                    exp_datum = self.all_datas["explanation"][datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)

                    user_id = self.user2id[exp_datum['reviewerID']]
                    target_id = self.item2id[exp_datum["asin"]]

                    slot_text = f" {self.sep_token} ".join([f"{self.user_token}_" + user_id,
                                                            f"{self.item_token}_" + target_id])
                    target_text = exp_datum["explanation"]
                    if task_id == "3-1":
                        input_text = group[prompt_id]

                    elif task_id == "3-2":
                        input_text = group[prompt_id].format(exp_datum["feature"])

                    elif task_id == "3-3":
                        input_text = group[prompt_id].format(exp_datum["overall"])

                    elif task_id == "3-4":
                        input_text = group[prompt_id].format(exp_datum["feature"], exp_datum["overall"])

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "review":
                    review_datum = self.all_datas["review"][datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)

                    user_id = self.user2id[review_datum['reviewerID']]

                    if task_id == "4-1":
                        slot_text = f"{self.user_token}_" + user_id
                        target_text = str(review_datum["overall"])
                        input_text = group[prompt_id].format(review_datum["reviewText"])

                    elif task_id == "4-2":
                        slot_text = f"{self.user_token}_" + user_id
                        if random.random() > 0.5:
                            input_text = group[prompt_id].format(review_datum["reviewText"], review_datum["overall"])
                            target_text = "yes"
                        else:
                            overall_candidates = [_ for _ in range(0 + 1, 5 + 1) if _ != int(review_datum['overall'])]
                            overall_idx = random.randint(0, len(overall_candidates) - 1)
                            input_text = group[prompt_id].format(review_datum["reviewText"],
                                                                 overall_candidates[overall_idx])
                            target_text = "no"

                    elif task_id == "4-3":
                        continue

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "traditional":
                    sequential_datum = self.all_datas["sequential"][datum_idx]
                    sequence = sequential_datum.split()
                    user_id = sequence[0]
                    user_desc = self.user_id2name[user_id]

                    if self.mode == 'train':
                        target_candidates = sequence[1:-2]
                        target_idx = random.randint(0, len(target_candidates) - 1)
                        target_item = target_candidates[target_idx]
                    elif self.mode == 'val':
                        target_item = sequence[-2]
                    elif self.mode == 'test':
                        target_item = sequence[-1]
                    else:
                        raise NotImplementedError
                    target_item = f"{self.item_token}_" + target_item

                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)

                    if task_id == "5-1":
                        rand_prob = random.random()
                        if rand_prob > 0.5:
                            slot_text = f" {self.sep_token} ".join([f"{self.user_token}_" + user_id, target_item])
                            target_text = "yes"
                            input_text = group[prompt_id]
                        else:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = 1
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                            candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                            slot_text = f" {self.sep_token} ".join(
                                [f"{self.user_token}_" + user_id, candidate_samples[0]])
                            target_text = "no"
                            input_text = group[prompt_id]

                    elif task_id == "5-2":
                        user_seq = self.user_items[user_id]
                        candidate_samples = []
                        candidate_num = 99  # random.randint(19, 99)
                        while len(candidate_samples) < candidate_num:
                            if self.sample_type == 'random':
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                            else:
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                              p=self.probability)
                            sample_ids = [str(item) for item in sample_ids if
                                          item not in user_seq and item not in candidate_samples]
                            candidate_samples.extend(sample_ids)
                        candidate_samples = candidate_samples[:candidate_num]
                        candidate_samples.append(target_item)
                        random.shuffle(candidate_samples)
                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(candidate_samples)])
                        target_text = target_item
                        input_text = group[prompt_id]

                    elif task_id == "5-3":
                        user_seq = self.user_items[user_id]
                        candidate_samples = []
                        candidate_num = 99  # random.randint(19, 99)
                        while len(candidate_samples) < candidate_num:
                            if self.sample_type == 'random':
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                            else:
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                              p=self.probability)
                            sample_ids = [str(item) for item in sample_ids if
                                          item not in user_seq and item not in candidate_samples]
                            candidate_samples.extend(sample_ids)
                        if random.random() > 0.5:
                            candidate_samples = candidate_samples[:candidate_num - 1]
                            candidate_samples.append(target_item)
                            random.shuffle(candidate_samples)
                            target_text = "yes"
                        else:
                            candidate_samples = candidate_samples[:candidate_num]
                            random.shuffle(candidate_samples)
                            target_text = "no"
                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(candidate_samples)])
                        input_text = group[prompt_id]

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "item_desc":
                    target_datum = self.meta_data[datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = self.prompts[task_name][task_id]
                    prompt_id = random.randint(0, len(group) - 1)
                    target_id = f"{self.item_token}_" + self.item2id[target_datum["business_id"]]

                    target_desc = self.__construct_item_desc(target_datum)

                    if task_id == "6-1":
                        slot_text = target_id
                        input_text = group[prompt_id].format(target_desc)

                        # negative slot samples
                        candidate_slot_samples = []
                        while len(candidate_slot_samples) < self.neg_samples_for_hfm:
                            neg_idx = random.sample(self.meta_data, self.neg_samples_for_hfm)
                            sample_ids = [sample["business_id"] for sample in neg_idx if
                                          sample["business_id"] != target_datum["business_id"] and sample[
                                              "business_id"] not in candidate_slot_samples]
                            candidate_slot_samples.extend(sample_ids)
                        candidate_slot_samples = [f"{self.item_token}_" + self.item2id[sample_id] for sample_id in
                                                  candidate_slot_samples]
                        candidate_slot_samples = candidate_slot_samples[:self.neg_samples_for_hfm - 1]
                        candidate_slot_samples.append(target_id)
                        random.shuffle(candidate_slot_samples)
                        neg_slot_labels = [int(sample == target_id) for sample in candidate_slot_samples]
                        assert sum(neg_slot_labels) == 1

                        # negative desc_samples
                        candidate_desc_samples = []
                        while len(candidate_desc_samples) < self.neg_samples_for_hfm:
                            neg_idx = random.sample(range(len(self.meta_data)), self.neg_samples_for_hfm)
                            neg_samples = [idx for idx in neg_idx if
                                           self.meta_data[idx]["business_id"] != target_datum["business_id"] and
                                           self.meta_data[idx]["business_id"] not in candidate_desc_samples]
                            candidate_desc_samples.extend(neg_samples)
                        candidate_desc_samples = [self.__construct_item_desc(self.meta_data[idx]) for idx in
                                                  candidate_desc_samples]
                        target_desc = self.__construct_item_desc(target_datum)
                        candidate_desc_samples = candidate_desc_samples[:self.neg_samples_for_hfm - 1]
                        candidate_desc_samples.append(target_desc)
                        random.shuffle(candidate_desc_samples)
                        neg_desc_labels = [int(desc == target_desc) for desc in candidate_desc_samples]
                        assert sum(neg_desc_labels) == 1

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "seq_desc":
                    sequential_datum = self.all_datas["sequential"][datum_idx]
                    sequence = sequential_datum.split()
                    user_id = sequence[0]
                    history_limit = 20

                    if self.mode == "train":
                        end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                        end_index = random.randint(0, len(end_candidates) - 1)
                        end_pos = end_candidates[end_index]
                        start_candidates = [_ for _ in range(1, min(4, end_pos))]
                        start_index = random.randint(0, len(start_candidates) - 1)
                        start_pos = start_candidates[start_index]
                        purchase_history = sequence[start_pos:end_pos + 1]
                        target_item = sequence[end_pos + 1]
                    elif self.mode == 'val':
                        purchase_history = sequence[1:-2]
                        target_item = sequence[-2]
                    elif self.mode == 'test':
                        purchase_history = sequence[1:-1]
                        target_item = sequence[-1]
                    else:
                        raise NotImplementedError

                    if len(purchase_history) > history_limit:
                        purchase_history = purchase_history[-history_limit:]

                    target_datum = self.meta_data[self.meta_dict[self.id2item[target_item]]]
                    purchase_history = [f"{self.item_token}_" + item for item in purchase_history]

                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = self.prompts[task_name][task_id]
                    prompt_id = random.randint(0, len(group) - 1)
                    target_id = f"{self.item_token}_" + self.item2id[target_datum["business_id"]]
                    target_desc = self.__construct_item_desc(target_datum)

                    if task_id == "7-1":
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history)])
                        input_text = group[prompt_id].format(target_desc)

                        # negative slot samples
                        candidate_slot_samples = []
                        while len(candidate_slot_samples) < self.neg_samples_for_hfm:
                            neg_idx = random.sample(range(len(self.all_datas["sequential"])), self.neg_samples_for_hfm)
                            for idx in neg_idx:
                                if idx == datum_idx:
                                    continue
                                neg_sequential_datum = self.all_datas["sequential"][idx]
                                neg_sequence = neg_sequential_datum.split(" ")
                                if len(neg_sequence) <= 2:
                                    continue
                                neg_user_id = f"{self.user_token}_" + neg_sequence[0]
                                purchase_len = min(history_limit, random.randint(2, len(neg_sequence) - 2))
                                neg_purchase_history = [f"{self.item_token}_" + it for it in
                                                        neg_sequence[1:purchase_len]]
                                neg_target_item = f"{self.item_token}_" + neg_sequence[purchase_len]
                                if neg_target_item == target_id:
                                    continue
                                candidate_slot_samples.append(
                                    f" {self.sep_token} ".join([neg_user_id, " ".join(neg_purchase_history)]))

                        candidate_slot_samples = candidate_slot_samples[:self.neg_samples_for_hfm - 1]
                        candidate_slot_samples.append(slot_text)
                        random.shuffle(candidate_slot_samples)
                        neg_slot_labels = [int(sample == slot_text) for sample in candidate_slot_samples]
                        assert sum(neg_slot_labels) == 1

                        # negative desc_samples
                        candidate_desc_samples = []
                        while len(candidate_desc_samples) < self.neg_samples_for_hfm:
                            neg_idx = random.sample(range(len(self.meta_data)), self.neg_samples_for_hfm)
                            neg_samples = [idx for idx in neg_idx if
                                           self.meta_data[idx]["business_id"] != target_datum["business_id"] and
                                           self.meta_data[idx]["business_id"] not in candidate_desc_samples]
                            candidate_desc_samples.extend(neg_samples)
                        candidate_desc_samples = [self.__construct_item_desc(self.meta_data[idx]) for idx in
                                                  candidate_desc_samples]
                        target_desc = self.__construct_item_desc(target_datum)
                        candidate_desc_samples = candidate_desc_samples[:self.neg_samples_for_hfm - 1]
                        candidate_desc_samples.append(target_desc)
                        random.shuffle(candidate_desc_samples)
                        neg_desc_labels = [int(desc == target_desc) for desc in candidate_desc_samples]
                        assert sum(neg_desc_labels) == 1

                    else:
                        raise NotImplementedError(task_name)

                data_entry["slot_text"] = slot_text
                data_entry["target_text"] = target_text
                data_entry["input_text"] = input_text
                data_entry["neg_desc_labels"] = neg_desc_labels
                data_entry["neg_slot_labels"] = neg_slot_labels
                data_entry["candidate_desc_samples"] = candidate_desc_samples
                data_entry["candidate_slot_samples"] = candidate_slot_samples
                assert task_id is not None
                data_entry["task_id"] = task_id
                task_dict[task_id].append(idx)
                all_datas.append(data_entry)
            if self.mode != "test":
                with open(all_datas_path, "wb") as fp:
                    pickle.dump([all_datas, task_dict], fp)
        else:
            if self.args.local_rank == 0:
                logging.info(f"loading dataset from {all_datas_path}..")
            with open(all_datas_path, "rb") as fp:
                all_datas, task_dict = pickle.load(fp)
        if not self.do_hfm:
            all_datas = list(filter(lambda x: x["task"] not in ["item_desc", "seq_desc"], all_datas))
        return all_datas, task_dict

    def get_user_items(self):
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.all_datas["sequential"]:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1

        all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        probability = [value / sum_value for value in count]
        user_items = user_items
        return all_item, probability, user_items

    def load_meta_data(self):
        datamaps = load_json(os.path.join(self.args.project_root, 'data', self.split, 'datamaps.json'))
        user2id = datamaps['user2id']
        item2id = datamaps['item2id']
        user_list = list(datamaps['user2id'].keys())
        item_list = list(datamaps['item2id'].keys())
        id2item = datamaps['id2item']

        user_id2name = load_pickle(os.path.join(self.args.project_root, 'data', self.split, 'user_id2name.pkl'))

        meta_data = load_pickle(os.path.join(self.args.project_root, 'data', self.split, 'meta_data.pkl'))
        user_data = load_pickle(os.path.join(self.args.project_root, 'data', self.split, 'user_data.pkl'))
        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item['business_id']] = i
        user_meta_dict = {}
        for j, user_meta_item in enumerate(user_data):
            user_meta_dict[user_meta_item['user_id']] = j

        return user2id, item2id, user_list, item_list, id2item, user_id2name, meta_data, user_data, meta_dict, user_meta_dict

    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.all_datas["rating"]) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                if sum([int(ind.split('-')[1]) == 1 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 2 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 3 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][3]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.all_datas["explanation"]) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.all_datas["review"]) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                if sum([int(ind.split('-')[1]) == 1 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 2 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 3 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length

            elif key == "item_desc":
                if self.do_hfm:
                    self.total_length += len(self.meta_data) * self.sample_numbers[key]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                    curr = self.total_length

            elif key == "seq_desc":
                if self.do_hfm:
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                    curr = self.total_length
            else:
                raise NotImplementedError

    def construct_prompt_pool(self, group_id):
        def search_group():
            for input_style, out_style_dict in self.prompt_groups.items():
                for out_style, groups in out_style_dict.items():
                    if group_id in groups:
                        return [input_style, out_style]

        input_style, out_style = search_group()
        neg_candidate_pool = []
        while len(neg_candidate_pool) < min(self.args.neg_samples_for_icl, 10):
            candidate_output_style = [outsty for outsty in self.prompt_groups[input_style].keys() if
                                      outsty != out_style]
            neg_output_style = random.sample(candidate_output_style, 1)[0]
            neg_group_id = random.sample(self.prompt_groups[input_style][neg_output_style], 1)[0]
            neg_candidate_pool.append(neg_group_id)
        neg_pool = defaultdict(int)
        for group_id in neg_candidate_pool:
            neg_pool[group_id] += 1

        pos_candidate_pool = self.prompt_groups[input_style][out_style]
        pos_group_id = random.sample(pos_candidate_pool, 1)[0]
        return neg_pool, pos_group_id

    def collate_fn(self, batch):
        # batch for rec
        batch_rec = list(filter(lambda x: x["task"] not in ["item_desc", "seq_desc"], batch))
        batch_entry_rec = defaultdict(list)
        batch_size_rec = len(batch_rec)

        if batch_size_rec != 0:
            max_slot_len_rec = max(entry['slot_length'] for entry in batch_rec)
            max_target_len_rec = max(entry['target_length'] for entry in batch_rec)
            max_input_len_rec = max(entry['input_length'] for entry in batch_rec)

            slot_ids_rec = torch.ones(batch_size_rec, max_slot_len_rec, dtype=torch.long) * self.pad_token_id
            target_ids_rec = torch.ones(batch_size_rec, max_target_len_rec, dtype=torch.long) * self.pad_token_id
            input_ids_rec = torch.ones(batch_size_rec, max_input_len_rec, dtype=torch.long) * self.pad_token_id

            for i, entry in enumerate(batch_rec):
                slot_ids_rec[i, :entry['slot_length']] = entry['slot_ids']
                target_ids_rec[i, :entry['target_length']] = entry['target_ids']
                input_ids_rec[i, :entry['input_length']] = entry['input_ids']

                for key, val in entry.items():
                    if isinstance(val, str):
                        batch_entry_rec[key].append(val)

            word_mask_rec = target_ids_rec != self.pad_token_id
            target_ids_rec[~word_mask_rec] = -100

            batch_entry_rec['slot_ids'] = slot_ids_rec
            batch_entry_rec['target_ids'] = target_ids_rec
            batch_entry_rec['input_ids'] = input_ids_rec

            # prompt contrastive learning
            if self.do_icl:
                pcl_labels = torch.LongTensor([entry["pcl_labels"] for entry in batch_rec])
                num_candidate_prompts = len(pcl_labels[0])
                prompt_ids, prompt_lens = [], []
                for data_entry in batch_rec:
                    prompt_ids.extend(data_entry["prompt_ids"])
                    prompt_lens.extend(data_entry["prompt_lens"])
                max_prompt_len = max(prompt_lens)
                prompt_ids_tensor = torch.ones(batch_size_rec * num_candidate_prompts, max_prompt_len,
                                               dtype=torch.long) * self.pad_token_id
                for leng, ids in zip(prompt_lens, prompt_ids):
                    prompt_ids_tensor[:, :leng] = ids
                batch_entry_rec["prompt_ids"] = prompt_ids_tensor
                batch_entry_rec["prompt_lens"] = prompt_lens
                batch_entry_rec["prompt_labels"] = pcl_labels

        batch_entry_pscl = defaultdict(list)
        batch_size_pscl = 0
        if self.do_hfm:
            batch_pscl = list(filter(lambda x: x["task"] in ["item_desc", "seq_desc"], batch))
            batch_size_pscl = len(batch_pscl)
            assert batch_size_pscl + batch_size_rec == len(batch)

            # processing candidate set
            if batch_size_pscl != 0:
                candidate_slot_ids, candidate_slot_lens = [], []
                candidate_desc_ids, candidate_desc_lens = [], []
                for entry_pscl in batch_pscl:
                    candidate_slot_ids.extend(entry_pscl["candidate_slot_ids"])
                    candidate_slot_lens.extend(entry_pscl["candidate_slot_length"])
                    candidate_desc_ids.extend(entry_pscl["candidate_desc_ids"])
                    candidate_desc_lens.extend(entry_pscl["candidate_desc_length"])
                max_slot_len = max(candidate_slot_lens)
                max_desc_len = max(candidate_desc_lens)
                candidate_slot_tensor = torch.ones(batch_size_pscl * self.args.neg_samples_for_hfm,
                                                   max_slot_len, dtype=torch.long) * self.pad_token_id
                candidate_desc_tensor = torch.ones(batch_size_pscl * self.args.neg_samples_for_hfm,
                                                   max_desc_len, dtype=torch.long) * self.pad_token_id
                for i, (slot_id, slot_len, desc_id, desc_len) in enumerate(
                        zip(candidate_slot_ids, candidate_slot_lens, candidate_desc_ids, candidate_desc_lens)):
                    candidate_slot_tensor[i, :slot_len] = slot_id
                    candidate_desc_tensor[i, :desc_len] = desc_id

                # processing input
                max_slot_len_pscl = max(entry['slot_length'] for entry in batch_pscl)
                max_input_len_pscl = max(entry['input_length'] for entry in batch_pscl)
                slot_ids_pscl = torch.ones(batch_size_pscl, max_slot_len_pscl,
                                           dtype=torch.long) * self.pad_token_id
                input_ids_pscl = torch.ones(batch_size_pscl, max_input_len_pscl,
                                            dtype=torch.long) * self.pad_token_id

                for i, entry in enumerate(batch_pscl):
                    slot_ids_pscl[i, :entry['slot_length']] = entry['slot_ids']
                    input_ids_pscl[i, :entry['input_length']] = entry['input_ids']

                    for key, val in entry.items():
                        if isinstance(val, str):
                            batch_entry_pscl[key].append(val)

                batch_entry_pscl["candidate_slot_ids"] = candidate_slot_tensor
                batch_entry_pscl["candidate_slot_lens"] = torch.LongTensor(candidate_slot_lens)
                batch_entry_pscl["candidate_desc_ids"] = candidate_desc_tensor
                batch_entry_pscl["candidate_desc_lens"] = torch.LongTensor(candidate_desc_lens)
                batch_entry_pscl['slot_ids'] = slot_ids_pscl
                batch_entry_pscl['input_ids'] = input_ids_pscl
                slot_labels = [b["neg_item_labels"] for b in batch_pscl]
                batch_entry_pscl["slot_labels"] = torch.FloatTensor(slot_labels)
                desc_labels = [b["neg_desc_labels"] for b in batch_pscl]
                batch_entry_pscl["desc_labels"] = torch.FloatTensor(desc_labels)

        batch_entry_rec["batch_size"] = batch_size_rec
        batch_entry_pscl["batch_size"] = batch_size_pscl
        return batch_entry_rec, batch_entry_pscl

    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((1.0 + 1.4) / 2), std=torch.tensor((1.4 - 1.0) / 4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((1.5 + 2.4) / 2), std=torch.tensor((2.4 - 1.5) / 4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((2.5 + 3.4) / 2), std=torch.tensor((3.4 - 2.5) / 4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((3.5 + 4.4) / 2), std=torch.tensor((4.4 - 3.5) / 4)).item(), 1)
            else:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((4.5 + 5.0) / 2), std=torch.tensor((5.0 - 4.5) / 4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])


class AmazonDataset(BaseDataset):
    def __init__(self, args, tokenizer, prompts, sample_numbers, task_list,
                 split, mode="train", sample_type="random"):
        super().__init__(args, tokenizer, prompts, sample_numbers, task_list,
                         split=split, mode=mode, sample_type=sample_type)

        self.all_datas = self.load_data()
        self.all_item, self.probability, self.user_items = self.get_user_items()
        if mode == "test":
            self.negative_samples = ReadLineFromFile(
                os.path.join(self.args.project_root, 'data', split, 'negative_samples.txt'))
        # metad data
        self.user2id, self.item2id, self.user_list, self.item_list, \
            self.id2item, self.user_id2name, self.meta_data, self.meta_dict = self.load_meta_data()

        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

        self.datas, self.task_dict = self.try_cache()
        assert len(self.datas) == self.total_length
        if self.args.local_rank == 0:
            logging.info("data source: {}".format(split.split(",")))
            logging.info(f"total {mode} number: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        batch_data = self.datas[idx]
        task_name = batch_data["task"]
        task_id = batch_data["task_id"]
        input_text = batch_data["input_text"]
        slot_text = batch_data["slot_text"]
        target_text = batch_data["target_text"]

        assert input_text is not None and slot_text is not None
        input_text = f"{self.cls_token} " + input_text
        input_ids = self.tokenizer.encode(input_text, padding=True, truncation=True,
                                          max_length=self.args.max_input_length)
        tokenized_input_text = self.tokenizer.tokenize(input_text)
        whole_word_ids_nl = self.calculate_whole_word_ids(tokenized_input_text, input_ids)

        slot_text = f"{self.cls_token} " + slot_text
        slot_ids = self.tokenizer.encode(slot_text, padding=True, truncation=True,
                                         max_length=self.args.max_input_length, )
        tokenized_slot_text = self.tokenizer.tokenize(slot_text)
        whole_word_ids_slot = self.calculate_whole_word_ids(tokenized_slot_text, slot_ids)
        visible_matrix = self.calculate_visible_matrix(tokenized_slot_text, whole_word_ids_slot)
        if target_text is not None:
            target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True,
                                               max_length=self.args.max_gen_length)
        else:
            target_ids = None

        candidate_desc_samples = batch_data["candidate_desc_samples"]
        candidate_slot_samples = batch_data["candidate_slot_samples"]

        pcl_labels, prompt_ids_list, prompt_pool, prompt_lens = None, None, None, None
        if task_name in ["item_desc", "seq_desc"]:
            candidate_desc_ids = [self.tokenizer.encode(sample, max_length=self.args.max_input_length,
                                                        truncation=True) for sample in
                                  candidate_desc_samples]
            candidate_desc_ids = [torch.LongTensor(desc_id) for desc_id in candidate_desc_ids]
            candidate_tokenized_desc_text = [self.tokenizer.tokenize(sample) for sample in candidate_desc_samples]
            candidate_whole_word_ids_desc = [self.calculate_whole_word_ids(desc_text, desc_id) for
                                             desc_text, desc_id in
                                             zip(candidate_tokenized_desc_text, candidate_desc_ids)]
            candidate_desc_lens = [len(desc) for desc in candidate_desc_ids]

            candidate_slot_ids = [self.tokenizer.encode(sample, max_length=self.args.max_input_length,
                                                        truncation=True) for sample in
                                  candidate_slot_samples]
            candidate_tokenized_slot_text = [self.tokenizer.tokenize(sample) for sample in candidate_slot_samples]
            candidate_whole_word_ids_slot = [self.calculate_whole_word_ids(slot_text, slot_id) for
                                             slot_text, slot_id in
                                             zip(candidate_tokenized_slot_text, candidate_slot_ids)]
            candidate_visible_matrix = [self.calculate_visible_matrix(slot_text, slot_whole_word_id) for
                                        slot_text, slot_whole_word_id in zip(candidate_tokenized_slot_text,
                                                                             candidate_whole_word_ids_slot)]

            candidate_slot_ids = [torch.LongTensor(item_id) for item_id in candidate_slot_ids]
            candidate_slot_lens = [len(item) for item in candidate_slot_ids]

        else:
            candidate_desc_ids, candidate_slot_ids = None, None
            candidate_desc_lens, candidate_slot_lens = 0, 0
            candidate_visible_matrix = None
            candidate_whole_word_ids_slot = None
            candidate_whole_word_ids_desc = None

        if self.do_icl:
            # datas for instruction contrastive learning
            neg_pool, pos_group_id = self.construct_prompt_pool(task_id)
            neg_sample_ids = []
            for neg_group_id, num in neg_pool.items():
                if neg_group_id not in self.task_dict.keys():
                    continue
                neg_sample_ids.extend(random.sample(self.task_dict[neg_group_id], num))

            prompt_pool = []
            for neg_id in neg_sample_ids:
                prompt_pool.append(self.datas[neg_id]["input_text"])

            pos_prompt_id = random.sample(self.task_dict[pos_group_id], 1)[0]
            while pos_prompt_id == idx:
                pos_prompt_id = random.sample(self.task_dict[pos_group_id], 1)[0]
            pos_prompt = self.datas[pos_prompt_id]["input_text"]
            prompt_pool.append(pos_prompt)
            random.shuffle(prompt_pool)
            prompt_ids_list = [torch.LongTensor(self.tokenizer.encode(t, max_length=self.args.max_input_length,
                                                                      truncation=True)) for t in prompt_pool]
            prompt_lens = [len(prompt) for prompt in prompt_ids_list]
            pcl_labels = [int(prompt == pos_prompt) for prompt in prompt_pool]
            assert sum(pcl_labels) == 1

        neg_desc_labels = batch_data["neg_desc_labels"]
        neg_slot_labels = batch_data["neg_slot_labels"]

        out_dict = {'input_ids': torch.LongTensor(input_ids),
                    'input_length': len(input_ids),
                    'whole_word_ids_nl': whole_word_ids_nl,
                    'target_ids': torch.LongTensor(target_ids) if target_ids is not None else None,
                    'target_length': len(target_ids) if target_ids is not None else 0,
                    "slot_ids": torch.LongTensor(slot_ids),
                    "slot_length": len(slot_ids),
                    'whole_word_ids_slot': whole_word_ids_slot,
                    "candidate_slot_ids": candidate_slot_ids,
                    "candidate_slot_length": candidate_slot_lens,
                    "candidate_whole_word_ids_slot": candidate_whole_word_ids_slot,
                    "candidate_desc_ids": candidate_desc_ids,
                    "candidate_desc_length": candidate_desc_lens,
                    "candidate_whole_word_ids_desc": candidate_whole_word_ids_desc,
                    'target_text': target_text,
                    "input_text": input_text,
                    'slot_text': slot_text,
                    "candidate_slot_text": candidate_slot_samples,
                    "candidate_desc_test": candidate_desc_samples,
                    "neg_desc_labels": neg_desc_labels,
                    "neg_item_labels": neg_slot_labels,
                    "pcl_labels": pcl_labels,
                    "prompt_ids": prompt_ids_list,
                    "prompt_text": prompt_pool,
                    "prompt_lens": prompt_lens,
                    'task': task_name}
        return out_dict

    def __construct_item_desc(self, item_datums):
        features = ["description", "title", "brand", "price"]
        # features = "description"
        desc = ""
        for key in features:
            if key in item_datums.keys():
                desc += f"{key}: {item_datums[key]}\n"
        return desc

    def try_cache(self):
        all_datas_path = os.path.join(self.args.project_root, "data", self.split,
                                      f"{self.mode}_" + self.args.all_data_file)
        if self.mode == "test" or not os.path.exists(all_datas_path):
            if self.args.local_rank == 0:
                logging.info(f"processing {self.mode} datas")

            all_datas = []
            task_dict = defaultdict(list)
            for idx, datum_info_idx in tqdm(enumerate(self.datum_info), total=len(self.datum_info)):
                data_entry = {}
                assert datum_info_idx[0] == idx
                if len(datum_info_idx) == 3:
                    task_name = datum_info_idx[1]
                    datum_idx = datum_info_idx[2]
                elif len(datum_info_idx) == 4:
                    task_name = datum_info_idx[1]
                    datum_idx = datum_info_idx[2]
                    task_id = datum_info_idx[3]
                else:
                    raise NotImplementedError
                data_entry["task"] = task_name
                data_entry["idx"] = idx
                data_entry["datum_idx"] = datum_info_idx[2]

                slot_text, target_text, input_text = None, None, None
                neg_desc_labels, neg_slot_labels = None, None
                candidate_desc_samples, candidate_slot_samples = None, None
                task_id = None

                if task_name == 'rating':
                    rating_datum = self.all_datas["rating"][datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = self.prompts["rating"][task_id]

                    user_id = f"{self.user_token}_" + self.user2id[rating_datum["reviewerID"]]
                    target_id = f"{self.item_token}_" + self.item2id[rating_datum["asin"]]
                    prompt_id = random.randint(0, len(group) - 1)
                    if task_id == "1-1":
                        slot_text = f" {self.sep_token} ".join([user_id, target_id])
                        target_text = self.gaussian_sampling(rating_datum)
                        target_text = str(target_text)
                        input_text = group[prompt_id]

                    elif task_id == "1-2":
                        slot_text = f" {self.sep_token} ".join([user_id, target_id])
                        rand_prob = random.random()
                        if rand_prob > 0.5:
                            target_text = "yes"
                            input_text = group[prompt_id].format(int(rating_datum["overall"]))
                        else:
                            target_text = "no"
                            overall_candidates = [_ for _ in range(0 + 1, 5 + 1) if _ != int(rating_datum['overall'])]
                            overall_idx = random.randint(0, len(overall_candidates) - 1)
                            input_text = group[prompt_id].format(overall_candidates[overall_idx])

                    elif task_id == "1-3":
                        slot_text = f" {self.sep_token} ".join([user_id, target_id])
                        if rating_datum["overall"] >= 4.0:
                            target_text = "yes"
                        else:
                            target_text = "no"
                        input_text = group[prompt_id]

                    else:
                        raise NotImplementedError("rating")

                elif task_name == "sequential":
                    sequential_datum = self.all_datas["sequential"][datum_idx]
                    sequence = sequential_datum.split()
                    user_id = sequence[0]
                    user_desc = self.user_id2name[user_id]
                    history_limit = 20

                    if self.mode == "train":
                        end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                        end_index = random.randint(0, len(end_candidates) - 1)
                        end_pos = end_candidates[end_index]
                        start_candidates = [_ for _ in range(1, min(4, end_pos))]
                        start_index = random.randint(0, len(start_candidates) - 1)
                        start_pos = start_candidates[start_index]
                        purchase_history = sequence[start_pos:end_pos + 1]
                        target_item = sequence[end_pos + 1]
                    elif self.mode == 'val':
                        purchase_history = sequence[1:-2]
                        target_item = sequence[-2]
                    elif self.mode == 'test':
                        purchase_history = sequence[1:-1]
                        target_item = sequence[-1]
                    else:
                        raise NotImplementedError

                    if len(purchase_history) > history_limit:
                        purchase_history = purchase_history[-history_limit:]
                    target_item = f"{self.item_token}_" + target_item
                    purchase_history = [f"{self.item_token}_" + item for item in purchase_history]

                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)
                    if task_id == "2-1":
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history)])
                        target_text = target_item
                        input_text = group[prompt_id]

                    elif task_id == "2-2":
                        if self.mode in ['train', 'val']:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = random.randint(79, 99)
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                        elif self.mode == 'test':
                            assert user_id == self.negative_samples[int(user_id) - 1].split(' ', 1)[0]
                            candidate_samples = self.negative_samples[int(user_id) - 1].split(' ', 1)[1].split(' ')
                        else:
                            raise NotImplementedError
                        candidate_samples.extend([target_item])
                        random.shuffle(candidate_samples)
                        rand_prob = random.random()

                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history), " ".join(candidate_samples)])
                        target_text = target_item
                        input_text = group[prompt_id]

                    elif task_id == "2-3":
                        rand_prob = random.random()
                        if rand_prob > 0.5:
                            slot_text = f" {self.sep_token} ".join(
                                [f"{self.user_token}_" + user_id, " ".join(purchase_history),
                                 target_item])
                            target_text = "yes"
                        else:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = 1
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                            candidate_samples = [f"{self.user_token}_" + sample for sample in candidate_samples]
                            slot_text = f" {self.sep_token} ".join(
                                [f"{self.user_token}_" + user_id, " ".join(purchase_history),
                                 candidate_samples[0]])
                            target_text = "no"
                        input_text = group[prompt_id]

                    elif task_id == "2-4":
                        if self.mode in ['train', 'val']:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = random.randint(79, 99)
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                        elif self.mode == 'test':
                            assert user_id == self.negative_samples[int(user_id) - 1].split(' ', 1)[0]
                            candidate_samples = self.negative_samples[int(user_id) - 1].split(' ', 1)[1].split(' ')
                        else:
                            raise NotImplementedError

                        if random.random() > 0.5:
                            candidate_samples.extend([target_item])
                            random.shuffle(candidate_samples)
                            target_text = "yes"
                        else:
                            target_text = "no"

                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history), " ".join(candidate_samples)])
                        input_text = group[prompt_id]

                    else:
                        raise NotImplementedError("sequential recommendation")

                elif task_name == "explanation":
                    exp_datum = self.all_datas["explanation"][datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)

                    user_id = self.user2id[exp_datum['reviewerID']]
                    target_id = self.item2id[exp_datum["asin"]]

                    slot_text = f" {self.sep_token} ".join([f"{self.user_token}_" + user_id,
                                                            f"{self.item_token}_" + target_id])
                    target_text = exp_datum["explanation"]
                    if task_id == "3-1":
                        input_text = group[prompt_id]

                    elif task_id == "3-2":
                        input_text = group[prompt_id].format(exp_datum["feature"])

                    elif task_id == "3-3":
                        input_text = group[prompt_id].format(exp_datum["overall"])

                    elif task_id == "3-4":
                        input_text = group[prompt_id].format(exp_datum["feature"], exp_datum["overall"])

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "review":
                    review_datum = self.all_datas["review"][datum_idx]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]
                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)

                    user_id = self.user2id[review_datum['reviewerID']]

                    if task_id == "4-1":
                        slot_text = f"{self.user_token}_" + user_id
                        target_text = str(review_datum["overall"])
                        input_text = group[prompt_id].format(review_datum["reviewText"])

                    elif task_id == "4-2":
                        slot_text = f"{self.user_token}_" + user_id
                        if random.random() > 0.5:
                            input_text = group[prompt_id].format(review_datum["reviewText"], review_datum["overall"])
                            target_text = "yes"
                        else:
                            overall_candidates = [_ for _ in range(0 + 1, 5 + 1) if _ != int(review_datum['overall'])]
                            overall_idx = random.randint(0, len(overall_candidates) - 1)
                            input_text = group[prompt_id].format(review_datum["reviewText"],
                                                                 overall_candidates[overall_idx])
                            target_text = "no"

                    elif task_id == "4-3":
                        # TODO
                        continue

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "traditional":
                    sequential_datum = self.all_datas["sequential"][datum_idx]
                    sequence = sequential_datum.split()
                    user_id = sequence[0]
                    user_desc = self.user_id2name[user_id]

                    if self.mode == 'train':
                        target_candidates = sequence[1:-2]
                        target_idx = random.randint(0, len(target_candidates) - 1)
                        target_item = target_candidates[target_idx]
                    elif self.mode == 'val':
                        target_item = sequence[-2]
                    elif self.mode == 'test':
                        target_item = sequence[-1]
                    else:
                        raise NotImplementedError
                    target_item = f"{self.item_token}_" + target_item

                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = task_candidates[task_id]
                    prompt_id = random.randint(0, len(group) - 1)

                    if task_id == "5-1":
                        rand_prob = random.random()
                        if rand_prob > 0.5:
                            slot_text = f" {self.sep_token} ".join([f"{self.user_token}_" + user_id, target_item])
                            target_text = "yes"
                            input_text = group[prompt_id]
                        else:
                            user_seq = self.user_items[user_id]
                            candidate_samples = []
                            candidate_num = 1
                            while len(candidate_samples) < candidate_num:
                                if self.sample_type == 'random':
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                                else:
                                    sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                                  p=self.probability)
                                sample_ids = [str(item) for item in sample_ids if
                                              item not in user_seq and item not in candidate_samples]
                                candidate_samples.extend(sample_ids)
                            candidate_samples = candidate_samples[:candidate_num]
                            candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                            slot_text = f" {self.sep_token} ".join(
                                [f"{self.user_token}_" + user_id, candidate_samples[0]])
                            target_text = "no"
                            input_text = group[prompt_id]

                    elif task_id == "5-2":
                        user_seq = self.user_items[user_id]
                        candidate_samples = []
                        candidate_num = 99  # random.randint(19, 99)
                        while len(candidate_samples) < candidate_num:
                            if self.sample_type == 'random':
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                            else:
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                              p=self.probability)
                            sample_ids = [str(item) for item in sample_ids if
                                          item not in user_seq and item not in candidate_samples]
                            candidate_samples.extend(sample_ids)
                        candidate_samples = candidate_samples[:candidate_num]
                        candidate_samples.append(target_item)
                        random.shuffle(candidate_samples)
                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(candidate_samples)])
                        target_text = target_item
                        input_text = group[prompt_id]

                    elif task_id == "5-3":
                        user_seq = self.user_items[user_id]
                        candidate_samples = []
                        candidate_num = 99  # random.randint(19, 99)
                        while len(candidate_samples) < candidate_num:
                            if self.sample_type == 'random':
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                            else:
                                sample_ids = np.random.choice(self.all_item, candidate_num, replace=False,
                                                              p=self.probability)
                            sample_ids = [str(item) for item in sample_ids if
                                          item not in user_seq and item not in candidate_samples]
                            candidate_samples.extend(sample_ids)
                        if random.random() > 0.5:
                            candidate_samples = candidate_samples[:candidate_num - 1]
                            candidate_samples.append(target_item)
                            random.shuffle(candidate_samples)
                            target_text = "yes"
                        else:
                            candidate_samples = candidate_samples[:candidate_num]
                            random.shuffle(candidate_samples)
                            target_text = "no"
                        candidate_samples = [f"{self.item_token}_" + sample for sample in candidate_samples]
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(candidate_samples)])
                        input_text = group[prompt_id]

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "item_desc":
                    target_datum = self.meta_data[self.meta_dict[self.item_list[datum_idx]]]
                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = self.prompts[task_name][task_id]
                    prompt_id = random.randint(0, len(group) - 1)
                    target_id = f"{self.item_token}_" + self.item2id[self.item_list[datum_idx]]
                    target_desc = self.__construct_item_desc(target_datum)

                    if task_id == "6-1":
                        slot_text = target_id
                        input_text = group[prompt_id].format(target_desc)

                        # negative slot samples
                        candidate_slot_samples = []
                        temp_target_id = self.item2id[self.item_list[datum_idx]]
                        while len(candidate_slot_samples) < self.neg_samples_for_hfm:
                            neg_idx = random.sample(self.item_list, self.neg_samples_for_hfm)
                            neg_idx = [self.item2id[idx] for idx in neg_idx]
                            neg_idx = list(
                                filter(lambda x: x != temp_target_id and x not in candidate_slot_samples, neg_idx))
                            candidate_slot_samples.extend(neg_idx)
                        candidate_slot_samples = [f"{self.item_token}_" + neg_id for neg_id in
                                                  candidate_slot_samples]
                        candidate_slot_samples = candidate_slot_samples[:self.neg_samples_for_hfm - 1]
                        candidate_slot_samples.append(target_id)
                        random.shuffle(candidate_slot_samples)
                        neg_slot_labels = [int(sample == target_id) for sample in candidate_slot_samples]
                        assert sum(neg_slot_labels) == 1

                        # negative desc_samples
                        candidate_desc_samples = []
                        while len(candidate_desc_samples) < self.neg_samples_for_hfm:
                            neg_samples = random.sample(self.meta_data, self.neg_samples_for_hfm)
                            neg_samples = list(
                                filter(lambda x: x != target_datum and x not in candidate_desc_samples, neg_samples))
                            candidate_desc_samples.extend(neg_samples)
                        candidate_desc_samples = [self.__construct_item_desc(sample) for sample in
                                                  candidate_desc_samples]
                        target_desc = self.__construct_item_desc(target_datum)
                        candidate_desc_samples = candidate_desc_samples[:self.neg_samples_for_hfm - 1]
                        candidate_desc_samples.append(target_desc)
                        random.shuffle(candidate_desc_samples)
                        neg_desc_labels = [int(desc == target_desc) for desc in candidate_desc_samples]
                        assert sum(neg_desc_labels) == 1

                    else:
                        raise NotImplementedError(task_name)

                elif task_name == "seq_desc":
                    sequential_datum = self.all_datas["sequential"][datum_idx]
                    sequence = sequential_datum.split()
                    user_id = sequence[0]
                    history_limit = 20

                    if self.mode == "train":
                        end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                        end_index = random.randint(0, len(end_candidates) - 1)
                        end_pos = end_candidates[end_index]
                        start_candidates = [_ for _ in range(1, min(4, end_pos))]
                        start_index = random.randint(0, len(start_candidates) - 1)
                        start_pos = start_candidates[start_index]
                        purchase_history = sequence[start_pos:end_pos + 1]
                        target_item = sequence[end_pos + 1]
                    elif self.mode == 'val':
                        purchase_history = sequence[1:-2]
                        target_item = sequence[-2]
                    elif self.mode == 'test':
                        purchase_history = sequence[1:-1]
                        target_item = sequence[-1]
                    else:
                        raise NotImplementedError

                    if len(purchase_history) > history_limit:
                        purchase_history = purchase_history[-history_limit:]

                    target_datum = self.meta_data[self.meta_dict[self.id2item[target_item]]]
                    purchase_history = [f"{self.item_token}_" + item for item in purchase_history]

                    task_candidates = self.prompts[task_name]
                    task_id = random.sample(self.task_list[task_name], 1)[0]

                    group = self.prompts[task_name][task_id]
                    prompt_id = random.randint(0, len(group) - 1)
                    target_id = f"{self.item_token}_" + target_item
                    target_desc = self.__construct_item_desc(target_datum)

                    if task_id == "7-1":
                        slot_text = f" {self.sep_token} ".join(
                            [f"{self.user_token}_" + user_id, " ".join(purchase_history)])
                        input_text = group[prompt_id].format(target_desc)

                        # negative slot samples
                        candidate_slot_samples = []
                        while len(candidate_slot_samples) < self.neg_samples_for_hfm:
                            neg_idx = random.sample(range(len(self.all_datas["sequential"])), self.neg_samples_for_hfm)
                            for idx in neg_idx:
                                if idx == datum_idx:
                                    continue
                                neg_sequential_datum = self.all_datas["sequential"][idx]
                                neg_sequence = neg_sequential_datum.split(" ")
                                if len(neg_sequence) <= 2:
                                    continue
                                neg_user_id = f"{self.user_token}_" + neg_sequence[0]
                                purchase_len = min(history_limit, random.randint(2, len(neg_sequence) - 2))
                                neg_purchase_history = [f"{self.item_token}_" + it for it in
                                                        neg_sequence[1:purchase_len]]
                                neg_target_item = f"{self.item_token}_" + neg_sequence[purchase_len]
                                if neg_target_item == target_id:
                                    continue
                                candidate_slot_samples.append(
                                    f" {self.sep_token} ".join([neg_user_id, " ".join(neg_purchase_history)]))

                        candidate_slot_samples = candidate_slot_samples[:self.neg_samples_for_hfm - 1]
                        candidate_slot_samples.append(slot_text)
                        random.shuffle(candidate_slot_samples)
                        neg_slot_labels = [int(sample == slot_text) for sample in candidate_slot_samples]
                        assert sum(neg_slot_labels) == 1

                        # negative desc_samples
                        candidate_desc_samples = []
                        while len(candidate_desc_samples) < self.neg_samples_for_hfm:
                            neg_samples = random.sample(self.meta_data, self.neg_samples_for_hfm)
                            neg_samples = list(
                                filter(lambda x: x != target_datum and x not in candidate_desc_samples, neg_samples))
                            candidate_desc_samples.extend(neg_samples)
                        candidate_desc_samples = [self.__construct_item_desc(sample) for sample in
                                                  candidate_desc_samples]
                        target_desc = self.__construct_item_desc(target_datum)
                        candidate_desc_samples = candidate_desc_samples[:self.neg_samples_for_hfm - 1]
                        candidate_desc_samples.append(target_desc)
                        random.shuffle(candidate_desc_samples)
                        neg_desc_labels = [int(desc == target_desc) for desc in candidate_desc_samples]
                        assert sum(neg_desc_labels) == 1

                    else:
                        raise NotImplementedError(task_name)

                data_entry["slot_text"] = slot_text
                data_entry["target_text"] = target_text
                data_entry["input_text"] = input_text
                data_entry["neg_desc_labels"] = neg_desc_labels
                data_entry["neg_slot_labels"] = neg_slot_labels
                data_entry["candidate_desc_samples"] = candidate_desc_samples
                data_entry["candidate_slot_samples"] = candidate_slot_samples
                assert task_id is not None
                data_entry["task_id"] = task_id
                task_dict[task_id].append(idx)
                all_datas.append(data_entry)
            if self.mode != "test":
                with open(all_datas_path, "wb") as fp:
                    pickle.dump([all_datas, task_dict], fp)

        else:
            if self.args.local_rank == 0:
                logging.info(f"loading dataset from {all_datas_path}..")
            with open(all_datas_path, "rb") as fp:
                all_datas, task_dict = pickle.load(fp)
            if not self.args.do_IDM:
                all_datas = list(filter(lambda x: x["task"] not in ["item_desc", "seq_desc"], all_datas))
        return all_datas, task_dict

    def load_data(self):
        all_datas = {}
        all_datas["review"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                       self.split, 'review_splits.pkl'))[self.mode]
        all_datas["explanation"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                            self.split, 'exp_splits.pkl'))[self.mode]
        if self.rating_augment:
            all_datas["rating"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                           self.split, 'rating_splits_augmented.pkl'))[self.mode]
        else:
            all_datas["rating"] = all_datas["review"]
        all_datas["sequential"] = ReadLineFromFile(os.path.join(self.args.project_root, 'data',
                                                                self.split, 'sequential_data.txt'))
        if self.mode == "test":
            all_datas["zero_exp_data"] = load_pickle(os.path.join(self.args.project_root, 'data',
                                                                  self.split, 'zeroshot_exp_splits.pkl'))
        return all_datas

    def get_user_items(self):
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.all_datas["sequential"]:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1

        all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        probability = [value / sum_value for value in count]
        user_items = user_items
        return all_item, probability, user_items

    def load_meta_data(self):
        datamaps = load_json(os.path.join(self.args.project_root, 'data', self.split, 'datamaps.json'))
        user2id = datamaps['user2id']
        item2id = datamaps['item2id']
        user_list = list(datamaps['user2id'].keys())
        item_list = list(datamaps['item2id'].keys())
        id2item = datamaps['id2item']

        user_id2name = load_pickle(os.path.join(self.args.project_root, 'data', self.split, 'user_id2name.pkl'))

        meta_data = []
        for meta in parse(os.path.join(self.args.project_root, 'data', self.split, 'meta.json.gz')):
            meta_data.append(meta)
        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item['asin']] = i

        return user2id, item2id, user_list, item_list, id2item, user_id2name, meta_data, meta_dict

    def construct_prompt_pool(self, group_id):
        def search_group():
            for input_style, out_style_dict in self.prompt_groups.items():
                for out_style, groups in out_style_dict.items():
                    if group_id in groups:
                        return [input_style, out_style]

        input_style, out_style = search_group()
        neg_candidate_pool = []
        while len(neg_candidate_pool) < min(self.args.neg_samples_for_icl, 10):
            candidate_output_style = [outsty for outsty in self.prompt_groups[input_style].keys() if
                                      outsty != out_style]
            neg_output_style = random.sample(candidate_output_style, 1)[0]
            neg_group_id = random.sample(self.prompt_groups[input_style][neg_output_style], 1)[0]
            neg_candidate_pool.append(neg_group_id)
        neg_pool = defaultdict(int)
        for group_id in neg_candidate_pool:
            neg_pool[group_id] += 1

        pos_candidate_pool = self.prompt_groups[input_style][out_style]
        pos_group_id = random.sample(pos_candidate_pool, 1)[0]
        return neg_pool, pos_group_id

    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.all_datas["rating"]) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                if sum([int(ind.split('-')[1]) == 1 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 2 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 3 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key][3]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.all_datas["explanation"]) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.all_datas["review"]) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                if sum([int(ind.split('-')[1]) == 1 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 2 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                if sum([int(ind.split('-')[1]) == 3 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length

            elif key == "item_desc":
                if self.do_hfm:
                    self.total_length += len(self.item_list) * self.sample_numbers[key]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                    curr = self.total_length

            elif key == "seq_desc":
                if self.do_hfm:
                    self.total_length += len(self.all_datas["sequential"]) * self.sample_numbers[key]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                    curr = self.total_length
            else:
                raise NotImplementedError

    def collate_fn(self, batch):
        # batch for rec
        batch_rec = list(filter(lambda x: x["task"] not in ["item_desc", "seq_desc"], batch))
        batch_entry_rec = defaultdict(list)
        batch_size_rec = len(batch_rec)

        if batch_size_rec != 0:
            max_slot_len_rec = max(entry['slot_length'] for entry in batch_rec)
            max_target_len_rec = max(entry['target_length'] for entry in batch_rec)
            max_input_len_rec = max(entry['input_length'] for entry in batch_rec)

            slot_ids_rec = torch.ones(batch_size_rec, max_slot_len_rec, dtype=torch.long) * self.pad_token_id
            target_ids_rec = torch.ones(batch_size_rec, max_target_len_rec, dtype=torch.long) * self.pad_token_id
            input_ids_rec = torch.ones(batch_size_rec, max_input_len_rec, dtype=torch.long) * self.pad_token_id

            for i, entry in enumerate(batch_rec):
                slot_ids_rec[i, :entry['slot_length']] = entry['slot_ids']
                target_ids_rec[i, :entry['target_length']] = entry['target_ids']
                input_ids_rec[i, :entry['input_length']] = entry['input_ids']

                for key, val in entry.items():
                    if isinstance(val, str):
                        batch_entry_rec[key].append(val)

            word_mask_rec = target_ids_rec != self.pad_token_id
            target_ids_rec[~word_mask_rec] = -100

            batch_entry_rec['slot_ids'] = slot_ids_rec
            batch_entry_rec['target_ids'] = target_ids_rec
            batch_entry_rec['input_ids'] = input_ids_rec

            # prompt contrastive learning
            if self.do_icl:
                pcl_labels = torch.LongTensor([entry["pcl_labels"] for entry in batch_rec])
                num_candidate_prompts = len(pcl_labels[0])
                prompt_ids, prompt_lens = [], []
                for data_entry in batch_rec:
                    prompt_ids.extend(data_entry["prompt_ids"])
                    prompt_lens.extend(data_entry["prompt_lens"])
                max_prompt_len = max(prompt_lens)
                prompt_ids_tensor = torch.ones(batch_size_rec * num_candidate_prompts, max_prompt_len,
                                               dtype=torch.long) * self.pad_token_id
                for leng, ids in zip(prompt_lens, prompt_ids):
                    prompt_ids_tensor[:, :leng] = ids
                batch_entry_rec["prompt_ids"] = prompt_ids_tensor
                batch_entry_rec["prompt_lens"] = prompt_lens
                batch_entry_rec["prompt_labels"] = pcl_labels

        batch_entry_icl = defaultdict(list)
        batch_size_hfm = 0
        if self.do_hfm:
            batch_hfm = list(filter(lambda x: x["task"] in ["item_desc", "seq_desc"], batch))
            batch_size_hfm = len(batch_hfm)
            assert batch_size_hfm + batch_size_rec == len(batch)

            # processing candidate set
            if batch_size_hfm != 0:
                candidate_slot_ids, candidate_slot_lens = [], []
                candidate_desc_ids, candidate_desc_lens = [], []
                for entry_hfm in batch_hfm:
                    candidate_slot_ids.extend(entry_hfm["candidate_slot_ids"])
                    candidate_slot_lens.extend(entry_hfm["candidate_slot_length"])
                    candidate_desc_ids.extend(entry_hfm["candidate_desc_ids"])
                    candidate_desc_lens.extend(entry_hfm["candidate_desc_length"])
                max_slot_len = max(candidate_slot_lens)
                max_desc_len = max(candidate_desc_lens)
                candidate_slot_tensor = torch.ones(batch_size_hfm * self.args.neg_samples_for_hfm,
                                                   max_slot_len, dtype=torch.long) * self.pad_token_id
                candidate_desc_tensor = torch.ones(batch_size_hfm * self.args.neg_samples_for_hfm,
                                                   max_desc_len, dtype=torch.long) * self.pad_token_id
                for i, (slot_id, slot_len, desc_id, desc_len) in enumerate(
                        zip(candidate_slot_ids, candidate_slot_lens, candidate_desc_ids, candidate_desc_lens)):
                    candidate_slot_tensor[i, :slot_len] = slot_id
                    candidate_desc_tensor[i, :desc_len] = desc_id

                # processing input
                max_slot_len_hfm = max(entry['slot_length'] for entry in batch_hfm)
                max_input_len_hfm = max(entry['input_length'] for entry in batch_hfm)
                slot_ids_hfm = torch.ones(batch_size_hfm, max_slot_len_hfm,
                                           dtype=torch.long) * self.pad_token_id
                input_ids_hfm = torch.ones(batch_size_hfm, max_input_len_hfm,
                                            dtype=torch.long) * self.pad_token_id

                for i, entry in enumerate(batch_hfm):
                    slot_ids_hfm[i, :entry['slot_length']] = entry['slot_ids']
                    input_ids_hfm[i, :entry['input_length']] = entry['input_ids']

                    for key, val in entry.items():
                        if isinstance(val, str):
                            batch_entry_icl[key].append(val)

                batch_entry_icl["candidate_slot_ids"] = candidate_slot_tensor
                batch_entry_icl["candidate_slot_lens"] = torch.LongTensor(candidate_slot_lens)
                batch_entry_icl["candidate_desc_ids"] = candidate_desc_tensor
                batch_entry_icl["candidate_desc_lens"] = torch.LongTensor(candidate_desc_lens)
                batch_entry_icl['slot_ids'] = slot_ids_hfm
                batch_entry_icl['input_ids'] = input_ids_hfm
                slot_labels = [b["neg_item_labels"] for b in batch_hfm]
                batch_entry_icl["slot_labels"] = torch.FloatTensor(slot_labels)
                desc_labels = [b["neg_desc_labels"] for b in batch_hfm]
                batch_entry_icl["desc_labels"] = torch.FloatTensor(desc_labels)

        batch_entry_rec["batch_size"] = batch_size_rec
        batch_entry_icl["batch_size"] = batch_size_hfm
        return batch_entry_rec, batch_entry_icl

    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((1.0 + 1.4) / 2), std=torch.tensor((1.4 - 1.0) / 4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((1.5 + 2.4) / 2), std=torch.tensor((2.4 - 1.5) / 4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((2.5 + 3.4) / 2), std=torch.tensor((3.4 - 2.5) / 4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((3.5 + 4.4) / 2), std=torch.tensor((4.4 - 3.5) / 4)).item(), 1)
            else:
                sampled_rating = round(
                    torch.normal(mean=torch.tensor((4.5 + 5.0) / 2), std=torch.tensor((5.0 - 4.5) / 4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])


class AmazonDatasetForP5(AmazonDataset):
    def __init__(self, args, tokenizer, prompts, sample_numbers, task_list,
                 split, mode="train", sample_type="random"):
        super().__init__(args, tokenizer, prompts, sample_numbers, task_list,
                         split, mode=mode, sample_type=sample_type)

    def __getitem__(self, idx):
        batch_data = self.datas[idx]
        task_name = batch_data["task"]
        task_id = batch_data["task_id"]
        nl_text = batch_data["input_text"]
        slot_text = batch_data["slot_text"]
        target_text = batch_data["target_text"]

        assert nl_text is not None and slot_text is not None
        input_text = f"{self.cls_token} {nl_text} {self.sep_token} {slot_text}"
        input_ids = self.tokenizer.encode(input_text, padding=True, truncation=True,
                                          max_length=self.args.max_input_length)
        tokenized_input_text = self.tokenizer.tokenize(input_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_input_text, input_ids)

        if target_text is not None:
            target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True,
                                               max_length=self.args.max_gen_length)
        else:
            target_ids = None

        candidate_desc_samples = batch_data["candidate_desc_samples"]
        candidate_slot_samples = batch_data["candidate_slot_samples"]

        pcl_labels, prompt_ids_list, prompt_pool, prompt_lens = None, None, None, None
        if task_name in ["item_desc", "seq_desc"]:
            candidate_desc_ids = [self.tokenizer.encode(sample, max_length=self.args.max_input_length,
                                                        truncation=True) for sample in
                                  candidate_desc_samples]
            candidate_desc_ids = [torch.LongTensor(desc_id) for desc_id in candidate_desc_ids]
            candidate_desc_lens = [len(desc) for desc in candidate_desc_ids]

            candidate_slot_ids = [self.tokenizer.encode(sample, max_length=self.args.max_input_length,
                                                        truncation=True) for sample in
                                  candidate_slot_samples]
            candidate_slot_ids = [torch.LongTensor(item_id) for item_id in candidate_slot_ids]
            candidate_slot_lens = [len(item) for item in candidate_slot_ids]

        else:
            candidate_desc_ids, candidate_slot_ids = None, None
            candidate_desc_lens, candidate_slot_lens = 0, 0

            if self.do_icl:
                # datas for contrastive instruction learning
                neg_pool, pos_group_id = self.construct_prompt_pool(task_id)
                neg_sample_ids = []
                for neg_group_id, num in neg_pool.items():
                    if neg_group_id not in self.task_dict.keys():
                        continue
                    neg_sample_ids.extend(random.sample(self.task_dict[neg_group_id], num))

                prompt_pool = []
                for neg_id in neg_sample_ids:
                    prompt_pool.append(self.datas[neg_id]["input_text"])

                pos_prompt_id = random.sample(self.task_dict[pos_group_id], 1)[0]
                while pos_prompt_id == idx:
                    pos_prompt_id = random.sample(self.task_dict[pos_group_id], 1)[0]
                pos_prompt = self.datas[pos_prompt_id]["input_text"]
                prompt_pool.append(pos_prompt)
                random.shuffle(prompt_pool)
                prompt_ids_list = [torch.LongTensor(self.tokenizer.encode(t, max_length=self.args.max_input_length,
                                                                          truncation=True)) for t in prompt_pool]
                prompt_lens = [len(prompt) for prompt in prompt_ids_list]
                pcl_labels = [int(prompt == pos_prompt) for prompt in prompt_pool]
                assert sum(pcl_labels) == 1

        neg_desc_labels = batch_data["neg_desc_labels"]
        neg_slot_labels = batch_data["neg_slot_labels"]

        out_dict = {'input_ids': torch.LongTensor(input_ids),
                    'input_length': len(input_ids),
                    'whole_word_ids': torch.LongTensor(whole_word_ids),
                    'target_ids': torch.LongTensor(target_ids) if target_ids is not None else None,
                    'target_length': len(target_ids) if target_ids is not None else 0,
                    "slot_ids": None,
                    "slot_length": None,
                    'whole_word_ids_slot': None,
                    "candidate_slot_ids": candidate_slot_ids,
                    "candidate_slot_length": candidate_slot_lens,
                    "candidate_desc_ids": candidate_desc_ids,
                    "candidate_desc_length": candidate_desc_lens,
                    'target_text': target_text,
                    "input_text": input_text,
                    'slot_text': slot_text,
                    "candidate_slot_text": candidate_slot_samples,
                    "candidate_desc_test": candidate_desc_samples,
                    "neg_desc_labels": neg_desc_labels,
                    "neg_item_labels": neg_slot_labels,
                    "pcl_labels": pcl_labels,
                    "prompt_ids": prompt_ids_list,
                    "prompt_text": prompt_pool,
                    "prompt_lens": prompt_lens,
                    'task': task_name}
        return out_dict

    def collate_fn(self, batch):
        # batch for rec
        batch_entry = defaultdict(list)
        batch_size = len(batch)

        max_target_len_rec = max(entry['target_length'] for entry in batch)
        max_input_len_rec = max(entry['input_length'] for entry in batch)

        target_ids_rec = torch.ones(batch_size, max_target_len_rec, dtype=torch.long) * self.pad_token_id
        input_ids_rec = torch.ones(batch_size, max_input_len_rec, dtype=torch.long) * self.pad_token_id
        whole_word_ids = torch.ones(batch_size, max_input_len_rec, dtype=torch.long) * self.pad_token_id

        for i, entry in enumerate(batch):
            target_ids_rec[i, :entry['target_length']] = entry['target_ids']
            input_ids_rec[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']

            for key, val in entry.items():
                if isinstance(val, str):
                    batch_entry[key].append(val)

        word_mask_rec = target_ids_rec != self.pad_token_id
        target_ids_rec[~word_mask_rec] = -100

        batch_entry['target_ids'] = target_ids_rec
        batch_entry['input_ids'] = input_ids_rec
        batch_entry['whole_word_ids'] = whole_word_ids

        batch_entry["batch_size"] = batch_size
        return batch_entry


def get_loader(args, tokenizer, template, sample_numbers, task_list, batch_size, mode):
    if args.data_name == "yelp":
        dataset = YelpDataset(args=args,
                              tokenizer=tokenizer,
                              prompts=template,
                              sample_numbers=sample_numbers,
                              task_list=task_list,
                              split=args.data_name,
                              mode=mode)
    elif args.data_name in ["beauty", "sports", "toys"]:
        data_class = AmazonDataset if args.model_class == "control_rec" else AmazonDatasetForP5
        dataset = data_class(args=args,
                             tokenizer=tokenizer,
                             prompts=template,
                             sample_numbers=sample_numbers,
                             task_list=task_list,
                             split=args.data_name,
                             mode=mode)
    else:
        raise NotImplementedError

    sampler = DistributedSampler(dataset) if args.distributed else None
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=args.n_gpu, pin_memory=True, sampler=sampler,
        collate_fn=dataset.collate_fn)
    return loader
