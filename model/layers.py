import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import gelu, get_attention_pad_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_prob)
        self.scale = 1 / (config.head_dim ** 0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        scores.masked_fill_(attention_mask, -1e9)
        # [batch_size, num_heads, sequence_length, sequence_length]
        attention_prob = nn.Softmax(dim=-1)(scores)
        attention_prob = self.dropout(attention_prob)

        # [batch_size, num_heads, sequence_length, head_dim]
        context = torch.matmul(attention_prob, value)

        return context, attention_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config

        self.W_q = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.W_k = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.W_v = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention(config)
        self.linear = nn.Linear(config.head_dim * config.num_heads, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        batch_size = query.size(0)

        # [batch_size, num_heads, sequence_length, head_dim]
        Q = self.W_q(query).view(batch_size, -1, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.config.num_heads, self.config.head_dim).transpose(1, 2)

        # [batch_size, num_heads, sequence_length, sequence_length]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.config.num_heads, 1, 1)

        # [batch_size, num_heads, sequence_length, head_dim]
        # [batch_size, num_heads, sequence_length, sequence_length]
        context, attention_prob = self.scaled_dot_product_attention(Q, K, V, attention_mask)

        # [batch_size, sequence_length, num_heads * head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.num_heads * self.config.head_dim)

        # [batch_size, sequence_length, hidden_dim]
        outputs = self.linear(context)
        outputs = self.dropout(outputs)

        return outputs, attention_prob


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config

        self.linear1 = nn.Linear(config.hidden_dim, config.feed_forward_dim)
        self.linear2 = nn.Linear(config.feed_forward_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor,
    ):
        outputs = self.linear1(inputs)
        outputs = gelu(outputs)
        outputs = self.linear2(outputs)
        return self.dropout(outputs)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config

        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.feed_forward = FeedForward(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        residual = inputs
        outputs, attention_prob = self.self_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
        )
        outputs = self.layer_norm1(outputs + residual)

        residual = outputs
        outputs = self.feed_forward(outputs)
        outputs = self.layer_norm2(outputs + residual)

        return outputs, attention_prob


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.enc_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.sequence_length + 1, config.hidden_dim)
        self.seg_emb = nn.Embedding(config.num_segments, config.hidden_dim)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])

    def forward(
        self,
        inputs: torch.Tensor,
        segments: torch.Tensor,
    ):
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)

        positions = torch.arange(
            sequence_length,
            device=inputs.device,
            dtype=inputs.dtype
        )
        positions = positions.expand(batch_size, sequence_length)
        positions = positions.contiguous() + 1

        # inputs 에서 값이 pad_token 인 위치들에 대응되는 positions 의 위치들의 값을 0으로 바꾼다.
        pos_mask = inputs.eq(self.config.pad_token)
        positions.masked_fill_(pos_mask, 0)

        # [batch_size, sequence_length, hidden_dim]
        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)

        # [batch_size, sequence_length, sequence_length]
        attention_mask = get_attention_pad_mask(inputs, inputs, self.config.pad_token)

        attention_probs = []
        for layer in self.layers:
            # outputs: [batch_size, sequence_length, hidden_dim]
            # attention_probs: [batch_size, num_head, sequence_length, sequence_length]
            outputs, attention_prob = layer(outputs, attention_mask)
            attention_probs.append(attention_prob)

        return outputs, attention_probs
