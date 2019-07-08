
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import pysnooper
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

@pysnooper.snoop('./attention.log')
def attention(query, key, value, mask=None, self_mask=False ,dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    assert query.dim() == 4 and key.dim()==4 and value.dim()==4  # x = [nbatch, head, kpairs, nbatch]
    d_k = query.size(-1) #the dim of the query
    # [nbatch, head, kpairs_q, d_k] [nbatch, head, d_k, kpairs_k] -> [nbatch, head, kpairs_q, kpairs_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    elif self_mask:
        scores = scores.masked_fill(scores==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # if dropout is not None: # drop out the attention is confusing
    #     p_attn = dropout(p_attn)
    #attn dot val = [nbatch, head, kpairs_q, kpairs_k] [nbatch, head, kpairs_v, d_k] = [nbatch, head, kpairs_q, d_k]
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    @pysnooper.snoop('./layernorm.log')
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None, self_mask=False):
        assert query.dim()==3 and key.dim()==3 and value.dim()==3
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x.view(-1, x.size(-1))).view(x.size(0),x.size(1), self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # q=[nbatch, kpairs, d_model]-> [nbatch*kpairs, d_model], qW=[nbatch*kpairs, d_model]-> [nbatch, kpairs, head, d_k] ->[nbatch, head, kpairs, d_k]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, self_mask = self_mask,
                                 dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        # x = [nbatch, head, kpairs, d_k] -> [nbatch, kpairs, head, d_k] -> [nbatch, kpairs, head*d_k]
        x = x.transpose(1, 2).contiguous() \
            .view(x.size(0), x.size(2), self.h * self.d_k)
        return self.linears[-1](x.view(-1, x.size(-1))).view(x.size(0), x.size(1), self.h*self.d_k)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)



class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class StarTransformer(nn.Module):
    def __init__(self, attn, norm):
        super(StarTransformer, self).__init__()
        self.attn = attn
        self.norm = norm

    def forward(self, inputs, t_rounds):
        # print("inputs:", inputs.shape)
        s_t = inputs.mean(dim=0).unsqueeze(0)

        d_model = inputs.size(1)
        inputs = torch.cat((torch.zeros(1, d_model), inputs, torch.zeros(1, d_model)), dim=0)
        H_t = inputs

        def get_ring(cur, max_num):
            return (cur-1 if cur-1>-1 else cur-1+max_num, cur, cur+1 if cur+1<max_num else cur+1-max_num)

        def update_ring(idx, oldH, s):
            (idx0, idx1, idx2) = get_ring(idx, oldH.size(0))
            C = torch.cat(
                (oldH[idx0], oldH[idx1], oldH[idx2], inputs[idx1], s[0])
            ).view(5, -1)
            # print("C_shape", C.shape)
            return self.norm(
                F.relu(
                    self.attn(s, C, C)
                )
            )

        for t in range(0, t_rounds, 1):
            H_t = torch.cat(
                list(
                    map( lambda idx: update_ring(idx, H_t, s_t), range(  H_t.size(0) ))
                ),
                dim=0
            )
            # print("H_t", H_t)
            s_t = self.attn(s_t, torch.cat((s_t, H_t), dim=0), torch.cat((s_t, H_t), dim=0))
            # print("s_t:", s_t)
            s_t = self.norm(F.relu(s_t))
        return s_t
