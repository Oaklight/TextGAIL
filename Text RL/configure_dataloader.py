import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from seq2seq_dataset import Seq2SeqDataset

# pylint:disable=no-member

DATA_DIR = "../../data"


class TextRLCollator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def sample_collate(self, batch):
        new_batch = {}
        # new_batch["source_text"] = [item["source_text"] for item in batch]
        # new_batch["target_text"] = [item["target_text"] for item in batch]
        new_batch["source_token_ids"] = pad_sequence(
            [item["source_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        new_batch["target_token_ids"] = pad_sequence(
            [item["target_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return new_batch


def lazy_collate(batch):
    return batch


class DataLoaderHandler:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def train_dataloader(self, config):

        train_filename = os.path.join(config.task.data_dir, "train.jsonl")
        train_dataset = Seq2SeqDataset(train_filename, self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.training.training.batch_size, shuffle=True, collate_fn=lazy_collate
        )
        return train_dataloader

    def valid_dataloader(self, config):
        valid_filename = os.path.join(config.task.data_dir, "val.jsonl")
        val_dataset = Seq2SeqDataset(valid_filename, self.tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=config.training.training.batch_size, collate_fn=lazy_collate)
        return val_dataloader
