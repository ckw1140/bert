import torch

from model.config import Config
from model.layers import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    Encoder,
)
from model.utils import get_attention_pad_mask


def test_scaled_dot_product_attention():
    config = Config.load("./tests/config.json")
    batch_size = 8

    query = torch.rand([batch_size, config.num_heads, config.sequence_length, config.head_dim])
    key = torch.rand([batch_size, config.num_heads, config.sequence_length, config.head_dim])
    value = torch.rand([batch_size, config.num_heads, config.sequence_length, config.head_dim])
    attention_mask = torch.zeros([batch_size, config.num_heads, config.sequence_length, config.sequence_length])

    scaled_dot_product_attention = ScaledDotProductAttention(config)
    context, attention_prob = scaled_dot_product_attention(query, key, value, attention_mask)

    assert context.size() == (batch_size, config.num_heads, config.sequence_length, config.head_dim)
    assert attention_prob.size() == (batch_size, config.num_heads, config.sequence_length, config.sequence_length)
    assert attention_prob.max() <= 1.0


def test_multi_head_attention():
    config = Config.load("./tests/config.json")
    batch_size = 8

    query = torch.rand([batch_size, config.sequence_length, config.hidden_dim])
    key = torch.rand([batch_size, config.sequence_length, config.hidden_dim])
    value = torch.rand([batch_size, config.sequence_length, config.hidden_dim])
    attention_mask = torch.zeros([batch_size, config.sequence_length, config.sequence_length])

    multi_head_attention = MultiHeadAttention(config)
    context, attention_prob = multi_head_attention(query, key, value, attention_mask)

    assert context.size() == (batch_size, config.sequence_length, config.hidden_dim)
    assert attention_prob.size() == (batch_size, config.num_heads, config.sequence_length, config.sequence_length)
    assert attention_prob.max() <= 1.0


def test_feed_forward():
    config = Config.load("./tests/config.json")
    batch_size = 8

    inputs = torch.rand([batch_size, config.sequence_length, config.hidden_dim])

    feed_forward = FeedForward(config)
    outputs = feed_forward(inputs)

    assert outputs.size() == (batch_size, config.sequence_length, config.hidden_dim)


def test_encoder_layer():
    config = Config.load("./tests/config.json")
    batch_size = 8

    inputs = torch.rand([batch_size, config.sequence_length, config.hidden_dim])
    attention_mask = torch.zeros([batch_size, config.sequence_length, config.sequence_length])

    encoder_layer = EncoderLayer(config)
    outputs, attention_prob = encoder_layer(inputs, attention_mask)

    assert outputs.size() == (batch_size, config.sequence_length, config.hidden_dim)
    assert attention_prob.size() == (batch_size, config.num_heads, config.sequence_length, config.sequence_length)


def test_encoder():
    config = Config.load("./tests/config.json")
    batch_size = 8

    inputs = torch.randint(config.vocab_size, (batch_size, config.sequence_length))
    segments = torch.randint(2, (batch_size, config.sequence_length))

    encoder = Encoder(config)
    outputs, attention_probs = encoder(inputs, segments)

    assert outputs.size() == (batch_size, config.sequence_length, config.hidden_dim)
    assert attention_probs[0].size() == (batch_size, config.num_heads, config.sequence_length, config.sequence_length)

