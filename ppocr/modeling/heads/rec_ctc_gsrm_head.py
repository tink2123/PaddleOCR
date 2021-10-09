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
import numpy as np

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F
from collections import OrderedDict

from .self_attention import WrapEncoder


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class GSRM(nn.Layer):
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, num_decoder_tus, hidden_dims):
        super(GSRM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.num_decoder_TUs = num_decoder_tus
        self.hidden_dims = hidden_dims

        self.fc0 = paddle.nn.Linear(
            in_features=in_channels, out_features=self.char_num)
        self.wrap_encoder0 = WrapEncoder(
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

        self.mul = lambda x: paddle.matmul(x=x,
                                           y=self.wrap_encoder0.prepare_decoder.emb0.weight,
                                           transpose_y=True)
        self.conv0 = paddle.nn.Conv1D(
            in_channels=80,
            out_channels=25,
            kernel_size=1)

    def forward(self, inputs, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2):
        # ===== GSRM Visual-to-semantic embedding block =====
        b, c, h, w = inputs.shape
        ### trans channel
        inputs = paddle.reshape(inputs, [-1, h*w, c])
        inputs = self.conv0(inputs)

        pvam_features = paddle.reshape(inputs, [-1, c])
        word_out = self.fc0(pvam_features)
        word_ids = paddle.argmax(F.softmax(word_out), axis=1)
        word_ids = paddle.reshape(x=word_ids, shape=[-1, 25, 1])


        #===== GSRM Semantic reasoning block =====
        """
        This module is achieved through bi-transformers,
        ngram_feature1 is the froward one, ngram_fetaure2 is the backward one
        """
        pad_idx = self.char_num

        word1 = paddle.cast(word_ids, "float32")
        word1 = F.pad(word1, [1, 0], value=1.0 * pad_idx, data_format="NLC")
        word1 = paddle.cast(word1, "int64")
        word1 = word1[:, :-1, :]
        word2 = word_ids

        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1] 
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]

        #print("enc_inputs_1:", [word1.shape, gsrm_word_pos.shape, gsrm_slf_attn_bias1.shape])   # [[128, 25, 1], [25, 1], [8, 25, 25]]


        gsrm_feature1 = self.wrap_encoder0(enc_inputs_1)
        gsrm_feature2 = self.wrap_encoder1(enc_inputs_2)

        gsrm_feature2 = F.pad(gsrm_feature2, [0, 1],
                              value=0.,
                              data_format="NLC")
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        gsrm_features = gsrm_feature1 + gsrm_feature2

        gsrm_out = self.mul(gsrm_features)

        b, t, c = gsrm_out.shape
        gsrm_out = paddle.reshape(gsrm_out, [-1, c])

        return gsrm_features, word_out, gsrm_out


class CTCHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 **kwargs):
        super(CTCHead, self).__init__()
        self.char_num = out_channels
        self.lstm = nn.LSTM(
            in_channels, 48, direction='bidirectional', num_layers=2)
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels)
        self.fc = nn.Linear(
            48*2,
            out_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        # add gsrm
        self.max_length=25
        self.num_heads = 8
        self.num_encoder_TUs=2
        self.num_decoder_TUs=4
        self.hidden_dims=512
        self.gsrm = GSRM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            num_decoder_tus=self.num_decoder_TUs,
            hidden_dims=self.hidden_dims)
        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0

    def forward(self, x, targets=None):
        others = targets[-4:]
        gsrm_word_pos = others[1]
        gsrm_slf_attn_bias1 = others[2]
        gsrm_slf_attn_bias2 = others[3]


        b,c,h,w = x.shape #(bs, 512, 1, 80)
        feature_dim = h*w
        
        gsrm_feature, word_out, gsrm_out = self.gsrm(
            x, paddle.to_tensor(gsrm_word_pos), paddle.to_tensor(gsrm_slf_attn_bias1),
            paddle.to_tensor(gsrm_slf_attn_bias2))

        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        x, _ = self.lstm(x)
        ctc_predicts = self.fc(x)
            
        if not self.training:
            ctc_predict = F.softmax(ctc_predicts, axis=2)
            predicts = {'predict':ctc_predict}

        else:
            predicts = OrderedDict([
                ('predict', ctc_predicts),
                ('word_out', word_out),
                ('gsrm_out', gsrm_out),
            ])
        return predicts
