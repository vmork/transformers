import warnings
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch 
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Int
from dataclasses import dataclass
from typing import Literal
from transformers import AutoTokenizer # type: ignore

from transformer.utils import Batch

@dataclass
class LanguageModelBatch(Batch):
    x: Int[Tensor, "B T"]
    y: Int[Tensor, "B T"] 

class TokenizedTextDataset(Dataset):
    def __init__(self, text_path: str, tokenizer: AutoTokenizer, n_ctx=512, train_ratio=0.9, split: Literal["train", "val"]="train"):
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Tokenizing text...")
        self.tokens = tokenizer.encode(text) # type: ignore
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens)}")
        
        split_idx = int(len(self.tokens) * train_ratio)
        if split == "train": self.tokens = self.tokens[:split_idx]
        elif split == "val": self.tokens = self.tokens[split_idx:]
        else: raise ValueError("split must be 'train' or 'val'")
        print(f"Tokens in split: {len(self.tokens)}")

        self.n_samples = max(0, len(self.tokens) - self.n_ctx)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        chunk = self.tokens[idx:idx+self.n_ctx+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def language_model_collator(batch) -> LanguageModelBatch:
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return LanguageModelBatch(xs, ys)