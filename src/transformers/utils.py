import torch
from torch import Tensor
from jaxtyping import Float
import torch.nn as nn
from dataclasses import dataclass, replace, fields
from typing import Callable, Any
import itertools
import abc
from collections.abc import Iterable

def flatten_list(x: Iterable[Iterable[Any]]) -> list[Any]:
    return list(itertools.chain.from_iterable(x))

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class Batch:
    def to(self, device: torch.device) -> "Batch":
        return apply_to_batch(self, lambda x: x.to(device))

def apply_to_batch(batch: Batch, f: Callable):
    return replace(batch, **{field.name: f(getattr(batch, field.name)) for field in fields(batch)})

@dataclass 
class ModelOutput:
    loss: Float[Tensor, ""]
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod 
    def get_output(self, batch: Batch) -> ModelOutput:
        pass
    