import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    # input_dim：输入的Q, K, V的特征维度
    def __init__(self, input_dim, output_dim, n_heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = output_dim
        self.n_heads = n_heads
        self.device = device

        # 强制 hid_dim 必须整除 h
        assert input_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(input_dim, output_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(input_dim, output_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(input_dim, output_dim)
        self.fc = nn.Linear(output_dim, output_dim)
        self.do = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([input_dim // n_heads]))

    def forward(self, query, key, value, mask=None):

        residual = query
        bsz = query.shape[0]
        Q = self.w_q(query).to(self.device)
        K = self.w_k(key).to(self.device)
        V = self.w_v(value).to(self.device)

        Q = Q.view(bsz, -1, self.n_heads, self.input_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.input_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.input_dim // self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2).to(self.device)) / self.scale.to(self.device)

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax

        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        x = x.view(bsz, -1, self.n_heads * (self.input_dim // self.n_heads))
        x = self.fc(x)

        # add & norm
        x += residual
        x = self.layer_norm(x)

        return x


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, input_dim, hid_dim, dropout):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hid_dim)
        self.w_2 = nn.Linear(hid_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.dropout(F.relu(self.w_1(x)))
        x = self.w_2(x)
        # add & norm
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, input_dim, output_dim, hid_dim, n_heads, dropout, device):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(input_dim, output_dim, n_heads, dropout, device)
        self.pos_ffn = PositionwiseFeedForward(input_dim, hid_dim, dropout)

    def forward(self, enc_input, slf_attn_mask=None):      # enc_input等于是MultiHeadAttention中输入的Q
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,  n_layers, input_dim, output_dim, hid_dim, n_heads, dropout, device):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(input_dim, output_dim, hid_dim, n_heads, dropout, device) for _ in range(n_layers)])

    def forward(self, enc_output):
        # Encoder Layers
        enc_output = self.dropout(enc_output)     #防止过拟合
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output



