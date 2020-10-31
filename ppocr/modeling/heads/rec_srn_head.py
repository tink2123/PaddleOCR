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

from paddle import nn, ParamAttr
from paddle.nn import functional as F
import paddle.fluid as fluid
import numpy as np
from self_attention import WrapEncoderForFeature
from .self_attention import WrapEncoder
gradient_clip = 10


class PVAM(nn.Layer):
    def __init__(self, in_channels, params):
        super(PVAM, self).__init__()
        self.char_num = params['char_num']
        self.max_length = params['max_text_length']

        self.num_heads = params['num_heads']
        self.num_encoder_TUs = params['num_encoder_TUs']
        self.num_decoder_TUs = params['num_decoder_TUs']
        self.hidden_dims = params['hidden_dims']
        # Transformer encoder
        t = 256
        c = 512
        self.wrap_encoder_for_feature = WrapEncoderForFeature(
            src_vocab_size=1,
            max_length=t,
            n_layer=self.num_encoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        # PVAM
        self.flatten0 = paddle.nn.Flatten(start_axis=0, stop_axis=1)
        self.fc0 = paddle.nn.Linear(
            in_features=in_channels,
            out_features=in_channels, )
        self.emb = paddle.nn.Embedding(
            num_embeddings=self.max_length, embedding_dim=in_channels)
        self.flatten1 = paddle.nn.Flatten(start_axis=0, stop_axis=2)
        self.fc1 = paddle.nn.Linear(
            in_features=in_channels, out_features=1, bias_attr=False)

    def forward(self, inputs, encoder_word_pos, gsrm_word_pos):
        b, c, h, w = inputs.shape
        print("inputs.shape:", inputs.shape)
        conv_features = paddle.reshape(inputs, shape=[-1, c, h * w])
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])
        # transformer encoder
        b, t, c = conv_features.shape
        #encoder_word_pos = others["encoder_word_pos"]
        #gsrm_word_pos = others["gsrm_word_pos"]

        enc_inputs = [conv_features, encoder_word_pos, None]
        print("conv_features:", conv_features.shape)
        print("encoder_word_pos:", encoder_word_pos.shape)
        word_features = self.wrap_encoder_for_feature(enc_inputs)

        # pvam
        b, t, c = word_features.shape
        word_features = self.flatten0(word_features)
        #print("after flatten:",word_features.shape)
        word_features = self.fc0(word_features)
        print("pvam word features shape:", word_features.shape)
        word_features_ = paddle.reshape(word_features, [-1, 1, t, c])
        print("reshape:", word_features.shape)
        word_features_ = paddle.tile(word_features_, [1, self.max_length, 1, 1])
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature_ = paddle.reshape(word_pos_feature,
                                           [-1, self.max_length, 1, c])
        word_pos_feature_ = paddle.tile(word_pos_feature_, [1, 1, t, 1])
        y = word_pos_feature_ + word_features_
        y = F.tanh(y)
        print("before flatten:", y.shape)
        #y = self.flatten1(y)
        print("after flatten:", y.shape)
        attention_weight = self.fc1(y)
        print("fc:", attention_weight.shape)
        attention_weight = paddle.reshape(
            attention_weight, shape=[-1, self.max_length, t])
        print("after reshape:", attention_weight.shape)
        attention_weight = F.softmax(attention_weight, axis=-1)
        pvam_features = paddle.matmul(attention_weight,
                                      word_features)  #[b, max_length, c]
        return pvam_features


class GSRM(nn.Layer):
    def __init__(self, in_channels, params):
        super(GSRM, self).__init__()
        self.char_num = params['char_num']
        self.max_length = params['max_text_length']

        self.num_heads = params['num_heads']
        self.num_encoder_TUs = params['num_encoder_TUs']
        self.num_decoder_TUs = params['num_decoder_TUs']
        self.hidden_dims = params['hidden_dims']

        self.fc0 = paddle.nn.Linear(
            in_features=in_channels, out_features=self.char_num)
        self.wrap_encoder0 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_encoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)
        self.wrap_encoder1 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

    def forward(self, inputs, others):
        # ===== GSRM Visual-to-semantic embedding block =====
        b, t, c = inputs.shape
        pvam_features = paddle.reshape(inputs, [-1, c])
        word_out = self.fc0(pvam_features)
        word_out = F.softmax(word_out)
        word_ids = paddle.argmax(word_out, axis=1)
        word_ids = paddle.reshape(x=word_ids, shape=[-1, t, 1])

        #===== GSRM Semantic reasoning block =====
        """
        This module is achieved through bi-transformers,
        ngram_feature1 is the froward one, ngram_fetaure2 is the backward one
        """
        pad_idx = self.char_num
        gsrm_word_pos = others['gsrm_word_pos']
        gsrm_slf_attn_bias1 = others['gsrm_slf_attn_bias1']
        gsrm_slf_attn_bias2 = others['gsrm_slf_attn_bias2']

        # prepare bi for gsrm, word1 for forward, word2 for backward
        word1 = paddle.cast(word_ids, "float32")
        word1 = F.pad(word1, [0, 0, 1, 0, 0, 0], value=1.0 * pad_idx)
        word1 = paddle.cast(word1, "int64")
        word1 = word1[:, :-1, :]
        word2 = word_ids

        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]


if __name__ == "__main__":

    import paddle
    paddle.disable_static()
    encoder_word_pos_np = np.random.random((1, 256, 1)).astype('int64')
    gsrm_word_pos_np = np.random.random((1, 25, 1)).astype('int64')
    encoder_word_pos = paddle.to_tensor(encoder_word_pos_np)
    gsrm_word_pos = paddle.to_tensor(gsrm_word_pos_np)
    others = {
        "encoder_word_pos": encoder_word_pos,
        "gsrm_word_pos": gsrm_word_pos
    }
    params = {
        "char_num": 38,
        "max_text_length": 25,
        "num_heads": 1,
        "num_encoder_TUs": 2,
        "num_decoder_TUs": 4,
        "hidden_dims": 512
    }
    pvam = PVAM(in_channels=512, params=params)
    data_np = np.random.random((1, 512, 8, 32)).astype('float32')
    data = paddle.to_tensor(data_np)
    output = pvam(data, encoder_word_pos, gsrm_word_pos)
    print("pvam shape:", output.shape)
