import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(inputs: torch.Tensor):
    """
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_gpt2.py
    를 참고하여 작성하였습니다.
    """
    cdf = 0.5 * (1.0 + torch.tanh((np.sqrt(2 / np.pi) * (inputs + 0.044715 * torch.pow(inputs, 3)))))
    return inputs * cdf


def get_attention_pad_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    pad_token: int,
):
    """
    attention_prob 에서 key 에 대응되는 위치가 pad_token 이라 masking 되야하는 위치에 True 값을,
    나머지 위치들에 대해서는 False 값을 갖는 Tensor를 반환합니다.
    """
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
