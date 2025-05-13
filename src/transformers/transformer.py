import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers.utils import Model

class TransformerEncoder(Model):
    def __init__(self, n_layers=12, n_head=12, d_emb=768, d_head=64, d_mlp=3072, p_dropout=0.1, masked_attention=False):
        super().__init__()
        assert d_head*n_head == d_emb, "Require d_head*n_head == d_emb"
        self.blocks = nn.ModuleList([
            TransformerBlock(n_head=n_head, d_emb=d_emb, d_mlp=d_mlp, d_head=d_head, p_dropout=p_dropout, masked_attention=masked_attention)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_emb)
    
    def forward(self, x: Float[Tensor, "B T d_emb"]) -> Float[Tensor, "B T d_emb"]:
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_head, d_emb, d_mlp, d_head, p_dropout, masked_attention=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_emb)
        self.attn = EfficientMultiHeadAttention(n_head=n_head, d_emb=d_emb, d_head=d_head, p_dropout=p_dropout, masked_attention=masked_attention)
        self.ln2 = nn.LayerNorm(d_emb)
        self.mlp = MLP(d_emb=d_emb, d_mlp=d_mlp, p_dropout=p_dropout)
        
    def forward(self, x: Float[Tensor, "B T d_emb"]) -> Float[Tensor, "B T d_emb"]:
        # pre-norm formulation
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class MLP(nn.Module):
    def __init__(self, d_emb, d_mlp, p_dropout):
        super().__init__()
        self.w1 = nn.Linear(d_emb, d_mlp)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(d_mlp, d_emb)
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self, x: Float[Tensor, "B T d_emb"]) -> Float[Tensor, "B T d_emb"]:
        return self.dropout(self.w2(self.gelu(self.w1(x))))
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_emb, d_head):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_emb=d_emb, d_head=d_head) for _ in range(n_head)])
        self.wo = nn.Linear(d_emb, d_emb) # == torch.cat([wo_1, ..., wo_12])
    
    def forward(self, x: Float[Tensor, "B T d_emb"]) -> Float[Tensor, "B T d_emb"]:
        x = torch.cat([h(x) for h in self.heads], dim=-1) # [B, T, d_head*n_head] = [B, T, d_emb]
        x = self.wo(x) # equivalent to `sum(wo_i * h_i(x) for i in range(n_layers))`
        return x

class AttentionHead(nn.Module):
    def __init__(self, d_emb, d_head):
        super().__init__()
        self.d_head = d_head
        self.wq = nn.Linear(d_emb, d_head, bias=False)
        self.wk = nn.Linear(d_emb, d_head, bias=False)
        self.wv = nn.Linear(d_emb, d_head, bias=False)
    
    def forward(self, x: Float[Tensor, "B T d_emb"]) -> Float[Tensor, "B T d_head"]:
        q = self.wq(x) # [B, T, d_head]
        k = self.wk(x) # [B, T, d_head]
        v = self.wv(x) # [B, T, d_head]
        A = (q @ k.transpose(-1, -2)) / self.d_head**0.5 # [B, T, T]
        A = F.softmax(A, dim=-1) # softmax along key dimension
        x = A @ v
        return x

class EfficientMultiHeadAttention(nn.Module):
    # Do everything at the same time!
    def __init__(self, n_head, d_emb, d_head, p_dropout, masked_attention=False):
        super().__init__()
        self.n_head, self.d_head, self.masked_attention = n_head, d_head, masked_attention
        # q,k,v matrices for all heads, stacked along last dim
        self.wqkv = nn.Linear(d_emb, 3*d_emb, bias=False)
        self.wo = nn.Linear(d_emb, d_emb)
        self.dropout_attn = nn.Dropout(p_dropout)
        self.dropout_resid = nn.Dropout(p_dropout)
    
    def forward(self, x: Float[Tensor, "B T d_emb"]) -> Float[Tensor, "B T d_emb"]:
        B, T, d_emb = x.shape
        assert d_emb == self.n_head*self.d_head
        
        # q,k,v: [B, T, d_emb = n_head*d_head]
        q, k, v = self.wqkv(x).split(d_emb, dim=-1) 
        # reshape to q,k,v: [B, n_head, T, d_head], so that we can compute all attention heads as batched mm
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2) # [B, n_head, T, d_head]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2) # [B, n_head, T, d_head]
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2) # [B, n_head, T, d_head]
        
        A = (q @ k.transpose(-1, -2)) / self.d_head**0.5 # [B, n_head, T, T]
        if self.masked_attention:
            mask = torch.tril(torch.ones((1, 1, T, T), device=x.device, dtype=torch.bool))
            A = A.masked_fill(~mask, float('-inf')) 
        A = A - A.amax(dim=-1, keepdim=True)  # subtract max for stability
        A = F.softmax(A, dim=-1)
        A = self.dropout_attn(A)
        
        y = A @ v # [B, n_head, T, d_head]
        y = y.transpose(1, 2).reshape(B, T, d_emb) # [B, T, d_emb=d_head*n_head]
        y = self.dropout_resid(self.wo(y)) # [B, T, d_emb]
        return y