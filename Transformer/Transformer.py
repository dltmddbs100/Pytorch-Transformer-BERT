import torch
import torch.nn as nn
import torch.nn.functional as F

import math, copy


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    '''
    input vector x는 (1,vocab)의 one-hot encoding vector,
    embedding W는 (vocab, d_model)로 x*W.transpose를 통해
    x의 vocab index에 해당하는 행을 추출
    '''
    def forward(self, x):
        # embedding vector에 sqrt(d_model)을 곱해 embedding vector값을 증가시킴
        # 이후 더해지는 position vector에 의해 embedding이 희석되는 것을
        # 방지하기 위함
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        pe_val = self.pe[:, :x.size(1)]
        pe_val.requires_grad = False
        
        x = x + pe_val
        x = self.dropout(x)
        return x        


def attention(query, key, value, mask=None, dropout=None):
    # query: [batch, h, n_seq, d_k]
    # key:   [nbatch, h, n_seq, d_k]
    # value: [nbatch, h, n_seq, d_v] (d_k=d_v)
    # 이 함수는 아래쪽 MultiHeadedAttention class에서 사용

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (nbatches, h, n_seq, n_seq)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn # [batch, h , n_seq, d_v] / [batch, h , n_seq, n_seq]

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
 
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value: [batch, n_seq, d_model]
        if mask is not None:
            # mask: [batch, 1, n_seq]
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        
        # 1) MultiHeadAttention format으로 d_model을 head 수로 분할 
        # self.linears는 요소 4개지만 (query, key, value)와 
        # 짝을 맞춰서 루프는 총 3번 돌아감
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # query, key, value: [batch, h, n_seq, d_k]
        
        # 2) Apply attention
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # x: [batch, h, n_seq, d_v],  self.attn: [batch, h, n_seq, n_seq]

        # 3) Concat all multiheads 
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k) # [batch, n_seq, h*d_k=d_model]

        # 4) final linear layer
        return self.linears[-1](x) # [batch, n_seq, d_model]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """sublayer dropout - residual connection - layer normalization"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        
        # self_attn: MultiHeadedAttention
        self.self_attn = self_attn

        # feed_forward: PositionwiseFeedForward
        self.feed_forward = feed_forward 
        
        # SublayerConnection이 2개
        self.sublayer = clones(SublayerConnection(size, dropout), 2) 
        self.size = size # d_model

    def forward(self, x, mask):
        # Attention - Dropout - Skip Connection - LayerNorm
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        return self.sublayer[1](x, self.feed_forward) # [batch, n_seq, d_model]


class Encoder(nn.Module):
    "Stack N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        
        self.layers = clones(layer, N) 
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # self.layers에는 EncoderLayer 여섯 개가 순차적으로 있음
        # N = 6
        for layer in self.layers: 
            x = layer(x, mask) 

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size

        # self_attn, src_attn: MultiHeadedAttention
        self.self_attn = self_attn
        self.src_attn = src_attn # cross attention
        self.feed_forward = feed_forward 
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        # memory = Encoder의 output
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # cross attention시에는 src_mask를 전달
        # key, value는 encoder output을 받음
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        
        return self.sublayer[2](x, self.feed_forward) # [batch, n_seq, d_model]


class Decoder(nn.Module):
    "Stack N layers."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return x # [batch, n_seq, d_model]


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()

        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1) # [batch, n_seq, vocab]


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # encoder embeding + positional encoding
        self.tgt_embed = tgt_embed # decoder embeding + positional encoding
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask), src_mask,
            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# Build Transformer
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
        src_vocab: input embedding vocab size
        tgt_vocab: output vocab size
        d_model: vector size
        d_ff: feed foward output channel size
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),           
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model

# Example
model = make_model(src_vocab = 10, tgt_vocab=10, N=2)

src= torch.randint(1,10,(1,10))
trg= torch.randint(1,10,(1,9))
src_mask=torch.tensor([[True, True, True, True, True, True, True, True, True, True]])
trg_mask=torch.tensor(
    [[[ True, False, False, False, False, False, False, False, False],
      [ True,  True, False, False, False, False, False, False, False],
      [ True,  True,  True, False, False, False, False, False, False],
      [ True,  True,  True,  True, False, False, False, False, False],
      [ True,  True,  True,  True,  True, False, False, False, False],
      [ True,  True,  True,  True,  True,  True, False, False, False],
      [ True,  True,  True,  True,  True,  True,  True, False, False],
      [ True,  True,  True,  True,  True,  True,  True,  True, False],
      [ True,  True,  True,  True,  True,  True,  True,  True,  True]]])

print(src.shape) # [batch, src_len]
print(trg.shape) # [batch, trg_len]
print(src_mask.shape) # [batch, src_len]
print(trg_mask.shape) # [batch, trg_len, trg_len]

model(src,trg,src_mask, trg_mask)