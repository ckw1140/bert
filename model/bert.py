import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Encoder


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.activation = torch.tanh

    def forward(
        self,
        inputs: torch.Tensor,
        segments: torch.Tensor,
    ):
        outputs, attention_probs = self.encoder(inputs, segments)

        # [batch_size, hidden_dim]
        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)

        return outputs, outputs_cls, attention_probs

    def save(self, epoch, loss, path):
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "state_dict": self.state_dict()
            },
            path,
        )
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class BERTPretrain(nn.Module):
    def __init__(self, config):
        super(BERTPretrain, self).__init__()
        self.config = config

        self.bert = BERT(config)

        # Classifier
        self.projection_cls = nn.Linear(config.hidden_dim, 2, bias=False)
        # lm
        self.projection_lm = nn.Linear(config.hidden_dim, config.vocab_size)
        self.projection_lm.weight = self.bert.encoder.enc_emb.weight

    def forward(
        self,
        inputs: torch.Tensor,
        segments: torch.Tensor,
    ):
        # [batch_size, sequence_length, hidden_dim]
        # [batch_size, hidden_dim]
        # [batch_size, num_heads, sequence_length, sequence_length] x num_layers
        outputs, outputs_cls, attention_probs = self.bert(inputs, segments)

        # [batch_size, 2]
        logits_cls = self.projection_cls(outputs_cls)
        # [batch_size, sequence_length, vocab_size]
        logits_lm = self.projection_lm(outputs)

        return logits_cls, logits_lm, attention_probs
