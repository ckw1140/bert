import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(inputs: torch.Tensor):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    :param inputs: float Tensor to perform activation.
    :returns: `inputs` with the GELU activation applied.
    .. note:: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_gpt2.py
    """
    cdf = 0.5 * (1.0 + torch.tanh((np.sqrt(2 / np.pi) * (inputs + 0.044715 * torch.pow(inputs, 3)))))
    return inputs * cdf


def get_attention_pad_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    pad_token: int,
):
    batch_size, query_length = query.size()
    batch_size, key_length = key.size()

    # attention_pad_mask.requires_grad=False
    # [batch_size, key_length]
    attention_pad_mask = key.data.eq(pad_token)
    # [batch_size, 1, key_length]
    attention_pad_mask = attention_pad_mask.unsqueeze(1)
    # [batch_size, query_length, key_length]
    attention_pad_mask = attention_pad_mask.expand(batch_size, query_length, key_length)
    return attention_pad_mask
