import torch

from model.bert import BERT, BERTPretrain
from model.config import Config

def test_bert():
    config = Config.load("./tests/config.json")
    batch_size = 8

    inputs = torch.randint(config.vocab_size, (batch_size, config.sequence_length))
    segments = torch.randint(2, (batch_size, config.sequence_length))

    bert = BERT(config)
    outputs, outputs_cls, attention_probs = bert(inputs, segments)

    assert outputs.size() == (batch_size, config.sequence_length, config.hidden_dim)
    assert outputs_cls.size() == (batch_size, config.hidden_dim)
    assert len(attention_probs) == config.num_layers
    assert attention_probs[0].size() == (batch_size, config.num_heads, config.sequence_length, config.sequence_length)
    assert attention_probs[0].max() <= 1.0


def test_bert_pretrain():
    config = Config.load("./tests/config.json")
    batch_size = 8

    inputs = torch.randint(config.vocab_size, (batch_size, config.sequence_length))
    segments = torch.randint(2, (batch_size, config.sequence_length))

    bert_pretrain = BERTPretrain(config)
    logits_cls, logits_lm, attention_probs = bert_pretrain(inputs, segments)

    assert logits_cls.size() == (batch_size, 2)
    assert logits_lm.size() == (batch_size, config.sequence_length, config.vocab_size)
    assert len(attention_probs) == config.num_layers
    assert attention_probs[0].size() == (batch_size, config.num_heads, config.sequence_length, config.sequence_length)
    assert attention_probs[0].max() <= 1.0
