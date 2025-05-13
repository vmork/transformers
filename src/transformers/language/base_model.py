import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from transformers.utils import Model
from transformers.transformer import TransformerEncoder

class LanguageModel(Model):
    def __init__(self, n_vocab=30522, n_layers=12, n_head=12, n_ctx=512, d_emb=768, d_head=64, d_mlp=3072, p_dropout=0.1, masked_attention=False):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, d_emb)
        self.pos_emb = nn.Embedding(n_ctx, d_emb)
        self.register_buffer("pos_idx", torch.arange(0, n_ctx))
        self.dropout = nn.Dropout(p_dropout)
        self.encoder = TransformerEncoder(n_layers=n_layers, n_head=n_head, d_emb=d_emb, d_head=d_head, d_mlp=d_mlp, p_dropout=p_dropout, masked_attention=masked_attention)
        
        self.head = nn.Linear(d_emb, n_vocab, bias=False)
        self.head.weight = self.tok_emb.weight
    
    def forward(self, x: Int[Tensor, "B T"]) -> Float[Tensor, "B T d_emb"]:
        B, T = x.shape
        xe = self.tok_emb(x) # [B, T, d_emb]
        xp = self.pos_emb(self.pos_idx) # [T, d_emb]
        x = xe + xp # [B, T, d_emb]
        x = self.dropout(x)
        x = self.encoder(x) # [B, T, d_emb]
        x = self.head(x)
        return x
        
