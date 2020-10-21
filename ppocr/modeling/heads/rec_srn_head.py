# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
from paddle import ParamAttr, nn
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import paddle.fluid as fluid
import numpy as np
from .self_attention.model import wrap_encoder
from .self_attention.model import wrap_encoder_forFeature
gradient_clip = 10

class SRN(nn.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SRN, self).__init__()

class PVAM(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(PVAM, self).__init__()
        self.word_features0 = WrapEncoderForFeature

class WrapEncoderForFeature(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(WrapEncoderForFeature, self).__init__()

class PrepareEncoder(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareEncoder, self).__init__()
        self.emb = paddle.nn.Embedding(
            num_embeddings=src_max_len,
            embedding_dim=src_emb_dim,
            weight_attr=paddle.ParamAttr(name=pos_enc_param_name, trainable=True))
        self.scale = paddle.scale(scale=src_emb_dim**0.5)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
    def forword(self, src_word, src_pos):
        src_word_emb = src_word
        src_word_emb = fluid.layers.cast(src_word_emb, 'float32')
        src_word_emb = self.scale(src_word_emb)
        src_pos_enc = self.emb(src_pos)
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = self.dropout(enc_input)
        else:
            out = enc_input
        return out

class PrepareDecoder(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate=0,
                 bos_idx=0,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareDecoder, self).__init__()
        self.emb = paddle.nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=src_emb_dim,
            weight_attr=paddle.ParamAttr(name=pos_enc_param_name,
                                         initializer=paddle.normal(mean=0,std=src_emb_dim**-0.5),
                                         trainable=True))
        self.scale = paddle.scale(scale=src_emb_dim ** 0.5)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
    def forword(self, src_word, src_pos):
        src_word_emb = self.emb(src_word)
        src_word_emb = self.scale(src_word_emb)
        src_pos_enc = self.emb(src_pos)
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = self.dropout(enc_input)
        else:
            out = enc_input
        return out

class Encoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

class Embedder(nn.Layer):
    """
    Word Embedding + Position Encoding
    """
    def __init__(self, vocab_size, emb_dim, bos_idx=0):
        super(Embedder, self).__init__()

        self.word_embedder = paddle.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            weight_attr=paddle.ParamAttr(initializer=paddle.normal(mean=0,std=src_emb_dim**-0.5),
                                         trainable=True))

    def forward(self, word):
        word_emb = self.word_embedder(word)
        return word_emb
