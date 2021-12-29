import torch
import torch.nn as nn
import torch.utils.checkpoint

import math
from typing import Dict, Optional, Tuple, List, Union
from functools import partial


class TokenEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self,
                x: torch.Tensor, # [batch, seq_len]
                transposed: bool = False):
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1)) # 추후 output embedding에서 사용
        else:
            return super().forward(x) # [batch, seq_len, embedding_dim]


class PositionalEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(self,
                              state_dict: Dict[str, torch.Tensor],
                              prefix: str,
                              *args,
                              **kwargs):
        weight = state_dict[f'{prefix}weight']

        # Reduce or expand the positional embedding matrix to increase or
        # decrease the total sequence length.
        if weight.size(0) < self.num_embeddings:
            weight = torch.cat((weight, self.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.num_embeddings:
            weight = weight[:self.num_embeddings]

        state_dict[f'{prefix}weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor: # [batch, seq_len]
        position = torch.arange(offset, offset + x.size(-1),
                                dtype=torch.long, device=x.device) # [offset + seq_len]
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x) # [batch, seq_len]

        return super().forward(position) # [batch, seq_len, embedding_dim]


Past = Tuple[torch.Tensor, torch.Tensor]

class BaseAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor, # [batch_size, heads, seq_len, dims/heads]
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # [batch_size, heads, seq_len, seq_len]

        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e4)
        x = self.dropout(x.softmax(-1))

        return torch.matmul(x, v) # [batch_size, heads, seq_len, dims/heads]


class MultiHeadAttention(BaseAttention):
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor, # [batch_size, seq_len, dims]
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
      
        # Split the tensors to multi-heads.
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads)) # [batch_size, seq_len, heads, dims/heads] 
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads)) 
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))

        q = q.transpose(-3, -2) # [batch_size, heads, seq_len, dims/heads] 
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Calculate multi-headed attentions and merge them into one.
        return (super().forward(q, k, v, mask) # [batch_size, heads, seq_len, dims/heads]
                .transpose(-3, -2) # [batch_size, seq_len, heads, dims/heads] 
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads))) # [batch_size, seq_len, dims] 


class AttentionLayer(nn.Module):
    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self,
                q: torch.Tensor, # [batch_size, seq_len, dims]
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None
                ):
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v) # [batch_size, seq_len, dims]

        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2) # [batch_size, past_len + seq_len, dims] 
            v = torch.cat((past[1], v), dim=-2)

        x = self.linear(self.attn(q, k, v, mask)) # [batch_size, seq_len, dims]
        return x, (k, v)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return x * self.sigmoid(x) 


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            nn.Linear(dims, dims * rate), # [batch_size, seq_len, dims*rate]
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims)) # [batch_size, seq_len, dims]


class PadMasking(nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0): # [batch, seq_len]
        is_pad = (x == self.pad_idx).unsqueeze(-2) # [batch, 1, seq_len]
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device) # [batch, 1, offset]

        mask = torch.cat((shifted, is_pad), dim=-1) # [batch, 1, seq_len + offset]
        return mask.expand(x.shape + mask.shape[-1:]) # [batch, seq_len, seq_len + offset]


class FutureMasking(nn.Module):
    def forward(self, x: torch.Tensor, offset: int = 0): # [batch, seq_len]
        seq_len = x.size(-1) # [seq_len]

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device) # [seq_len, seq_len + offset]
        future = future.triu(offset + 1) # [seq_len, seq_len + offset]

        mask = future.view((1,) * (x.ndim - 1) + future.size()) # [1, seq_len, seq_len + offset]
        return mask.expand(x.shape + mask.shape[-1:]) # [batch, seq_len, seq_len + offset]


class TransformerLayer(nn.Module):
    def __init__(self,
                 heads: int,
                 dims: int,
                 rate: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = nn.LayerNorm(dims)
        self.ln_ff = nn.LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                ):
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask) # [batch_size, seq_len, dims], [batch_size, past_len + seq_len, dims]

        x = x + a # add
        x = x + self.ff(self.ln_ff(x)) # [batch_size, seq_len, dims] 

        return x if self.training else (x, past)


class Transformer(nn.Module):
    def __init__(self,
                 layers: int,
                 pad_idx: int,
                 words: int,
                 seq_len: int,
                 heads: int,
                 dims: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        self.positional_embedding = PositionalEmbedding(seq_len, dims)
        self.token_embedding = TokenEmbedding(words, dims)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(heads, dims, rate, dropout)
            for _ in range(layers)])
        self.ln_head = nn.LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding and positional embedding layers.
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint,
                                      transformer)

            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)
        x = self.token_embedding(x, transposed=True)

        return x if self.training else (x, present)


# Example
src=torch.randint(10,(16,256))

GPT2=Transformer(layers=6, pad_idx=0, words=10000, seq_len=512, heads=6, dims=786)

logits=GPT2(src) # [batch, seq_len, vocab_size]
output_tokens=torch.argmax(nn.Softmax(-1)(logits),-1)

output_tokens.shape # [batch, seq_len]