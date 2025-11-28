### REPORDUCE 0.77


# ruff: noqa: D102 D103 D107 D101 F841


import copy
import math
from typing import List
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# from mamba_ssm.ops.triton.layernorm import RMSNorm
from torch import Tensor
from torch import einsum
from torch.nn import functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# bidirectional cross attention - have two sequences attend to each other with 1 attention step


class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        context_dim=None,
        dropout=0.0,
        talking_heads=False,
        prenorm=False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(self, x, context, mask=None, context_mask=None, return_attn=False, rel_pos_bias=None):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)
        qk, context_qk, v, context_v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (qk, context_qk, v, context_v)
        )
        sim = einsum("b h i d, b h j d -> b h i j", qk, context_qk) * self.scale
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device=device, dtype=torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device=device, dtype=torch.bool))

            attn_mask = rearrange(mask, "b i -> b 1 i 1") * rearrange(context_mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        context_attn = sim.softmax(dim=-2)
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, context_v)
        context_out = einsum("b h j i, b h j d -> b h i d", context_attn, v)
        out, context_out = map(lambda t: rearrange(t, "b h n d -> b n (h d)"), (out, context_out))
        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out


def get_emb(sin_inp):
    """Embedding for one dimension with sin and cos intertwined.

    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, max_len=1000):  # Add max_len parameter
        """Positional Encoding for 1D Tensors.

        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param max_len: Maximum sequence length to precompute.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute positional encoding for max_len
        pos_x = torch.arange(max_len).type(inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((max_len, channels)).type(inv_freq.type())
        emb[:, :channels] = emb_x
        self.register_buffer("cached_penc", emb)

    def forward(self, tensor):
        """Positional Encoding for 1D Tensors.

        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch).
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        batch_size, x, orig_ch = tensor.shape
        if x > self.cached_penc.shape[0]:
            raise ValueError(f"Sequence length {x} exceeds max_len {self.cached_penc.shape[0]}")

        return self.cached_penc[None, :x, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncoding(nn.Module):
    """Encoding the position of the token in the sequence.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Args.

        x: Tensor, shape [seq_len, batch_size, embedding_dim].

        """
        # x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """Positional Encoding for 2D Tensors.

        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """args.

        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class FeedForward(nn.Module):
    """Modularized feed forward layer.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, d_ff=2048, dropout=0.1, output=None):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        if output is None:
            self.linear_2 = nn.Linear(d_ff, d_model)
        else:
            self.linear_2 = nn.Linear(d_ff, output)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    """Modularized layer normalisation.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


class RBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(RBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
        self.running_mean = torch.zeros(1, num_features, 1)
        self.running_var = torch.ones(1, num_features, 1)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size(1), self.num_features, 1, device=x.device)
        x = x.permute(1, 0, 2).contiguous()
        batch_mean = x.mean([0, 1], keepdim=True)
        batch_var = x.var([0, 1], keepdim=True)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        output, hidden = self.rnn(x, hidden)
        output = output * self.gamma + self.beta
        output = output.permute(1, 0, 2).contiguous()
        return output, hidden


class Conv1DBlock(nn.Module):
    """Module class for 1D convolutional block.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels: int,
        out_channels_list: List[int],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        down_smaple: bool = True,
        dropout: Optional[float] = None,
        view: bool = False,
        dilation: List[int] = 1,
        bias: bool = False,
        normalization_end: bool = False,
    ):
        super(Conv1DBlock, self).__init__()
        if not isinstance(stride, list):
            stride = [stride]
        if not isinstance(dilation, list):
            dilation = [dilation]
        if len(dilation) != len(out_channels_list):
            dilation = dilation * len(out_channels_list)
        if len(stride) != len(out_channels_list):
            stride = stride * len(out_channels_list)
        layers = []
        for index, (out_channels, in_stride) in enumerate(zip(out_channels_list, stride)):
            layers.extend(
                (
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=in_stride,
                        padding=padding,
                        dilation=dilation[index],
                        bias=bias,
                    ),
                    # nn.AdaptiveMaxPool1d(out_channels)
                )
            )
            if normalization_end:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            if down_smaple:
                layers.append(nn.MaxPool1d(2))
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        # if not normalization_end:
        layers.append(nn.MaxPool1d(2))
        layers.append(nn.BatchNorm1d(out_channels_list[-1]))
        layers.append(nn.AdaptiveMaxPool1d(out_channels_list[-1]))
        self.conv_layers = nn.Sequential(*layers)
        if view:
            print(self.conv_layers)

    def forward(self, x):
        return self.conv_layers(x)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, device=None):
        super(MultiHeadAttentionLayer, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query_fc = nn.Linear(d_model, d_model)
        self.key_fc = nn.Linear(d_model, d_model)
        self.value_fc = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # / self.scale onny error so next line
        scores = torch.div(scores, self.scale)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_scores = torch.matmul(attn, V)

        attn_concat = attn_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(F.gelu(attn_concat))


class CustomMultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, k_d_model, v_d_model, n_heads, dropout=0.1, device=None):
        super(CustomMultiHeadAttentionLayer, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query_fc = nn.Linear(d_model, d_model)
        self.key_fc = nn.Linear(k_d_model, d_model)
        self.value_fc = nn.Linear(v_d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))  
        scores = torch.div(scores, self.scale)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_scores = torch.matmul(attn, V)

        attn_concat = attn_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(F.gelu(attn_concat))


class MultiLayerTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1, device=None, out_dim=None) -> None:
        super(MultiLayerTransformer, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]).to(
            device
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]).to(device)
        # self.norms = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(n_layers)]).to(device)

        out_dim = d_model if out_dim is None else out_dim
        self.fc_out = nn.Linear(d_model, out_dim)

    def forward(self, src):
        out = src
        for layer, norm in zip(self.layers, self.norms):
            residual = out
            out = layer(out, out, out)
            out = norm(out + residual)
        return self.fc_out(F.leaky_relu(out))


class CrossMultiLayerTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1, device=None, out_dim=None) -> None:
        super(CrossMultiLayerTransformer, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]).to(
            device
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]).to(device)
        out_dim = d_model if out_dim is None else out_dim
        self.fc_out = nn.Linear(d_model, out_dim)

    def forward(self, query, key, value):
        out = query
        for layer, norm in zip(self.layers, self.norms):
            residual = out
            out = layer(out, key, value)
            out = norm(out + residual)
        return self.fc_out(F.leaky_relu(out))


class CCrossMultiLayerTransformer(nn.Module):
    def __init__(
        self, d_model, n_heads, n_layers, k_d_model, v_d_model, dropout=0.1, device=None, out_dim=None
    ) -> None:
        super(CCrossMultiLayerTransformer, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList(
            [CustomMultiHeadAttentionLayer(d_model, k_d_model, v_d_model, n_heads, dropout) for _ in range(n_layers)]
        ).to(device)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]).to(device)
        out_dim = d_model if out_dim is None else out_dim
        self.fc_out = nn.Linear(d_model, out_dim)

    def forward(self, query, key, value):
        out = query
        for layer, norm in zip(self.layers, self.norms):
            residual = out
            out = layer(out, key, value)
            out = norm(out + residual)
        return self.fc_out(F.leaky_relu(out))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == "relu":
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == "tanh":
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == "none":
            self.dense_output_nonlinearity = lambda x: x

    def forward(self, x):
        if self.N_dense == 0:
            return x

        for i in range(len(self.linears) - 1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))