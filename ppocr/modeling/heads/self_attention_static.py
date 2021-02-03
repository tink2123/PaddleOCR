# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from paddle.nn import Transformer
from paddle.nn import TransformerEncoder
from paddle.nn import TransformerEncoderLayer
gradient_clip = 10


class WrapEncoderForFeature(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_idx=0):
        super(WrapEncoderForFeature, self).__init__()

        self.prepare_encoder = PrepareEncoder(
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            bos_idx=bos_idx,
            word_emb_param_name="src_word_emb_table")

        self.encoder_layer = TransformerEncoderLayer(
            d_model,
            n_head,
            d_inner_hid,
            prepostprocess_dropout,
            activation="relu")
        self.encoder = TransformerEncoder(self.encoder_layer, n_layer)

    def forward(self, enc_inputs):
        conv_features, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_encoder(conv_features, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class WrapEncoder(nn.Layer):
    """
    embedder + encoder
    """

    def __init__(self,
                 src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_idx=0):
        super(WrapEncoder, self).__init__()

        self.prepare_decoder = PrepareDecoder(
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            bos_idx=bos_idx)
        self.encoder_layer = TransformerEncoderLayer(
            d_model,
            n_head,
            d_inner_hid,
            prepostprocess_dropout,
            activation="relu")
        self.encoder = TransformerEncoder(self.encoder_layer, n_layer)

    def forward(self, enc_inputs):
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self.prepare_decoder(src_word, src_pos)
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


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
        self.src_emb_dim = src_emb_dim
        self.src_max_len = src_max_len
        self.emb = paddle.nn.Embedding(
            num_embeddings=self.src_max_len,
            embedding_dim=self.src_emb_dim,
            sparse=True)
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word_emb = src_word
        src_word_emb = fluid.layers.cast(src_word_emb, 'float32')
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        src_pos = paddle.squeeze(src_pos, axis=-1)
        src_pos_enc = self.emb(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(
                x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
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
        self.src_emb_dim = src_emb_dim
        """
        self.emb0 = Embedding(num_embeddings=src_vocab_size,
                              embedding_dim=src_emb_dim)
        """
        self.emb0 = paddle.nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=self.src_emb_dim,
            weight_attr=paddle.ParamAttr(
                name=word_emb_param_name,
                initializer=nn.initializer.Normal(0., src_emb_dim**-0.5)))
        self.emb1 = paddle.nn.Embedding(
            num_embeddings=src_max_len,
            embedding_dim=self.src_emb_dim,
            weight_attr=paddle.ParamAttr(name=pos_enc_param_name))
        self.dropout_rate = dropout_rate

    def forward(self, src_word, src_pos):
        src_word = fluid.layers.cast(src_word, 'int64')
        src_word = paddle.squeeze(src_word, axis=-1)
        src_word_emb = self.emb0(src_word)
        src_word_emb = paddle.scale(x=src_word_emb, scale=self.src_emb_dim**0.5)
        src_pos = paddle.squeeze(src_pos, axis=-1)
        src_pos_enc = self.emb1(src_pos)
        src_pos_enc.stop_gradient = True
        enc_input = src_word_emb + src_pos_enc
        if self.dropout_rate:
            out = F.dropout(
                x=enc_input, p=self.dropout_rate, mode="downscale_in_infer")
        else:
            out = enc_input
        return out

