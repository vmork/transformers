import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from dataclasses import dataclass
from transformers import AutoTokenizer # type: ignore

from transformer.utils import Model, ModelOutput, get_device
from transformer.transformer import TransformerEncoder
from transformer.language.data import LanguageModelBatch

@dataclass 
class LanguageModelOutput(ModelOutput):
    logits: Float[Tensor, "B T C"]
    loss: Float[Tensor, ""]

class LanguageModel(Model):
    def __init__(self, tokenizer: AutoTokenizer, n_layers=12, n_head=12, n_ctx=512, d_emb=768, d_head=64, d_mlp=3072, p_dropout=0.1, masked_attention=True):
        super().__init__()
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        self.n_vocab = tokenizer.vocab_size # type: ignore
        self.tok_emb = nn.Embedding(self.n_vocab, d_emb)
        self.pos_emb = nn.Embedding(n_ctx, d_emb)
        self.register_buffer("pos_idx", torch.arange(0, n_ctx))
        self.dropout = nn.Dropout(p_dropout)
        self.encoder = TransformerEncoder(n_layers=n_layers, n_head=n_head, d_emb=d_emb, d_head=d_head, d_mlp=d_mlp, p_dropout=p_dropout, masked_attention=masked_attention)
        
        self.head = nn.Linear(d_emb, self.n_vocab, bias=False)
        self.head.weight = self.tok_emb.weight # weight tying (https://paperswithcode.com/method/weight-tying)
        
        # important, since the head uses the same weight as the embedding and nn.Embedding's are initialized
        # with N(0, 1), this is too large for the head (that takes layernormed input) and will lead to high variance logits 
        # so we have decrease the standard deviation
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
    
    def forward(self, x: Int[Tensor, "B T"]) -> Float[Tensor, "B T C"]:
        B, T = x.shape
        xe = self.tok_emb(x) # [B, T, d_emb]
        xp = self.pos_emb(self.pos_idx[:T]) # [T, d_emb] # type: ignore
        x = xe + xp # [B, T, d_emb]
        x = self.dropout(x)
        x = self.encoder(x) # [B, T, d_emb]
        x = self.head(x)
        return x
    
    def get_output(self, batch: LanguageModelBatch) -> LanguageModelOutput: # type: ignore
        logits = self.forward(batch.x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.y.view(-1))
        return LanguageModelOutput(loss, logits)

    def generate(self, prompt: str, max_new_tokens: int=100, temperature: float=1.0) -> str:
        device = get_device()
        tokens = self.tokenizer.encode(prompt) # type: ignore
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        result_tokens = tokens.copy()
        for _ in range(max_new_tokens):
            if context.size(1) > self.n_ctx: context = context[:, -self.n_ctx:]
            logits = self.forward(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            result_tokens.append(next_token.item())
            context = torch.cat((context, next_token), dim=1)
        return self.tokenizer.decode(result_tokens) # type: ignore

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = LanguageModel(tokenizer)
    print(model.generate("Hello there!"))