from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    window_size: int


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, token_ids):
        pass


# class RotaryEmbedding(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#     def build_frequencies(self):
#         pass
#
#     def rotate_half(self, x):
#         pass
#
#     def forward(self, q, k):
#         pass


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

    def project_q(self, x):
        pass

    def project_k(self, x):
        pass

    def project_v(self, x):
        pass

    def split_heads(self, x):
        pass

    def merge_heads(self, x):
        pass

    def build_causal_mask(self, seq_len):
        pass

    def build_sliding_window_mask(self, seq_len):
        pass

    def apply_mask(self, scores):
        pass

    def scaled_dot_product_attention(self, q, k, v):
        pass

    def forward(self, x):
        pass


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        pass


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

    def forward(self, x):
        pass


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, token_ids):
        pass


class LanguageModelHead(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        pass


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, token_ids):
        pass

    def generate(self, token_ids, max_new_tokens):
        pass
