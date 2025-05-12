from typing import Any, Generator
from jaxtyping import Float, Int 
from torch import Tensor
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

from transformers.utils import Batch, ModelOutput

@dataclass
class CIFARBatch(Batch):
    x: Float[Tensor, "B 3 32 32"]
    y: Int[Tensor, "B"]

@dataclass 
class CIFAROutput(ModelOutput):
    loss: Float[Tensor, ""]
    logits: Float[Tensor, "B C"]

class CIFARDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        
    def __iter__(self) -> Generator[CIFARBatch, Any, None]: # type: ignore
        iterator = super().__iter__()
        for x, y in iterator:
            yield CIFARBatch(x=x, y=y)