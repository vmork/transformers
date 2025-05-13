from jaxtyping import Float 
import torch.nn as nn
import torch
from torch import Tensor

from transformers.vision.base_model import VisionModel
from transformers.transformer import TransformerEncoder

class EmbeddingLayer(nn.Module):
    """
    1. Takes a `(B, C, W, H)` image, splits it up into `N` non-overlapping `(C, P, P)` patches, \\
    where `N = W*H//P^2`, flattens them into `C*P^2` vectors and embeds them all to `D`-dimensional vectors.
    
    2. Inserts a class-embedding vector before the other embeddings.
    
    3. Adds 1D position embeddings `p_0, ..., p_N` to the patch embeddings.
        
    Returns a `(B, N+1, D)` tensor of embedding vectors.
    """
    def __init__(self, C: int, W: int, H: int, patch_size: int, d_emb: int):
        super().__init__()
        
        self.W, self.H = W, H
        self.P = patch_size
        self.C = C
        self.D = d_emb
                
        if self.W % self.P != 0 or self.H % self.P != 0:
            raise ValueError("`width` and `height` need to be divisible by `patch_size`")
        
        self.N = self.W*self.H // self.P**2
        
        self.patch_emb = nn.Linear(self.C*self.P**2, self.D)
        self.class_emb = nn.Parameter(torch.zeros((1, 1, self.D)))
        self.pos_emb = nn.Embedding(self.N+1, self.D)
        self.register_buffer("pos_idx", torch.arange(self.N+1))
    
    def forward(self, x: Float[Tensor, "B C W H"]) -> Float[Tensor, "B N+1 D"]:
        if x.ndim != 4: raise ValueError("Expected x to be a (B, C, W, H)-tensor")
        B, C, W, H = x.shape 
        if (C, W, H) != (self.C, self.W, self.H): raise ValueError(f"Expected x to be a (B, {self.C}, {self.W}, {self.H})-tensor")
        P = self.P
        
        ps = x.unfold(2, P, P).unfold(3, P, P) # [B,C,  W//P,H//P,  P,P]
        ps = ps.permute(0, 2, 3, 1, 4, 5)      # [B,    W//P,H//P,  C,P,P]
        ps = ps.flatten(1, 2)                  # [B,    N,          C,P,P]
        ps = ps.flatten(2)                     # [B,    N,          C*P*P]

        xe = self.patch_emb(ps)                # [B, N, D]
        xc = self.class_emb.expand(B, -1, -1)  # [B, 1, D]
        xp = self.pos_emb(self.pos_idx)        # [N+1, D]
        
        x = torch.cat([xc, xe], dim=1) # [B, N+1, D]
        x = x + xp                     # [B, N+1, D]

        return x

class VisionTransformer(VisionModel):
    def __init__(self, n_classes, C: int=3, W: int=32, H: int=32,  patch_size: int=4,
                 d_emb: int=768, n_layers=12, n_head=12, d_head=64, d_mlp=3072, p_dropout=0.1):
        super().__init__()
        self.embed = EmbeddingLayer(C=C, W=W, H=H, patch_size=patch_size, d_emb=d_emb)
        self.dropout_embed = nn.Dropout(p_dropout)
        self.encoder = TransformerEncoder(n_layers=n_layers, n_head=n_head, d_emb=d_emb, d_head=d_head, d_mlp=d_mlp, p_dropout=p_dropout)
        self.classifier = nn.Linear(d_emb, n_classes)
    
    def forward(self, x: Float[Tensor, "B C W H"]):
        x = self.embed(x)                    # [B, N+1, D]
        x = self.dropout_embed(x)
        x = self.encoder(x)                  # [B, N+1, D]
        logits = self.classifier(x[:, 0, :]) # [B, n_classes]
        return logits    

if __name__ == "__main__":
    vit = VisionTransformer(C=3, W=32, H=32, patch_size=4, n_classes=10)
    x = torch.randn((16, 3, 32, 32))
    y = vit(x)
    print(y.shape, y.mean(), y.var(), y)