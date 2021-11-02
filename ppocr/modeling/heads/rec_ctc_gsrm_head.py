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
from .self_attention import WrapEncoderForFeature


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class PVAM(nn.Layer):
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, hidden_dims):
        super(PVAM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.hidden_dims = hidden_dims
        # Transformer encoder
        t = 80
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
        # self.flatten0 = paddle.nn.Flatten(start_axis=0, stop_axis=1)
        # self.fc0 = paddle.nn.Linear(
        #     in_features=in_channels,
        #     out_features=in_channels, )
        # self.emb = paddle.nn.Embedding(
        #     num_embeddings=self.max_length, embedding_dim=in_channels)
        # self.flatten1 = paddle.nn.Flatten(start_axis=0, stop_axis=2)
        # self.fc1 = paddle.nn.Linear(
        #     in_features=in_channels, out_features=1, bias_attr=False)

        self.conv0 = paddle.nn.Conv1D(
            in_channels=80, out_channels=50, kernel_size=1)

    def forward(self, inputs, encoder_word_pos, gsrm_word_pos):
        b, c, h, w = inputs.shape
        conv_features = paddle.reshape(inputs, shape=[-1, c, h * w])
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])
        # transformer encoder
        b, t, c = conv_features.shape

        enc_inputs = [conv_features, encoder_word_pos, None]
        word_features = self.wrap_encoder_for_feature(enc_inputs)
        word_features = self.conv0(word_features)

        # pvam
        # b, t, c = word_features.shape
        # word_features = self.fc0(word_features)
        # word_features_ = paddle.reshape(word_features, [-1, 1, t, c])
        # word_features_ = paddle.tile(word_features_, [1, self.max_length, 1, 1])
        # word_pos_feature = self.emb(gsrm_word_pos)
        # word_pos_feature_ = paddle.reshape(word_pos_feature,
        #                                    [-1, self.max_length, 1, c])
        # word_pos_feature_ = paddle.tile(word_pos_feature_, [1, 1, t, 1])
        # # print("word_pos_feature_:{}, word_features_:{}".format(word_pos_feature_.shape, word_features_.shape))
        # y = word_pos_feature_ + word_features_
        # y = F.tanh(y)
        # attention_weight = self.fc1(y)
        # attention_weight = paddle.reshape(
        #     attention_weight, shape=[-1, self.max_length, t])
        # attention_weight = F.softmax(attention_weight, axis=-1)
        # pvam_features = paddle.matmul(attention_weight,
        #                               word_features)  #[b, max_length, c]
        return word_features


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
        # self.conv0 = paddle.nn.Conv1D(
        #     in_channels=80,
        #     out_channels=50,
        #     kernel_size=1)

    def forward(self, inputs, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2):
        # ===== GSRM Visual-to-semantic embedding block =====
        # b, c, h, w = inputs.shape
        # ### trans channel
        # inputs = paddle.reshape(inputs, [-1, h*w, c])
        # inputs = self.conv0(inputs)

        b, t, c = inputs.shape

        pvam_features = paddle.reshape(inputs, [-1, c])  # [50, 512]
        word_out = self.fc0(pvam_features)
        word_ids = paddle.argmax(F.softmax(word_out), axis=1)
        word_ids = paddle.reshape(x=word_ids, shape=[-1, 50, 1])

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

        # print("enc_inputs_1:", [word1.shape, gsrm_word_pos.shape, gsrm_slf_attn_bias1.shape])   # [[128, 25, 1], [25, 1], [8, 25, 25]]

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

        #return gsrm_features, word_out, gsrm_out
        return gsrm_features, gsrm_out


class VSFD(nn.Layer):
    def __init__(self, in_channels=512, pvam_ch=512, char_num=38):
        super(VSFD, self).__init__()
        self.char_num = char_num
        self.fc0 = paddle.nn.Linear(
            in_features=in_channels * 2, out_features=pvam_ch)
        self.fc1 = paddle.nn.Linear(
            in_features=pvam_ch, out_features=self.char_num)

    def forward(self, pvam_feature, gsrm_feature):
        b, t, c1 = pvam_feature.shape
        b, t, c2 = gsrm_feature.shape
        combine_feature_ = paddle.concat([pvam_feature, gsrm_feature], axis=2)
        img_comb_feature_ = paddle.reshape(
            combine_feature_, shape=[-1, c1 + c2])
        img_comb_feature_map = self.fc0(img_comb_feature_)
        img_comb_feature_map = F.sigmoid(img_comb_feature_map)
        img_comb_feature_map = paddle.reshape(
            img_comb_feature_map, shape=[-1, t, c1])
        combine_feature = img_comb_feature_map * pvam_feature + (
            1.0 - img_comb_feature_map) * gsrm_feature
        img_comb_feature = paddle.reshape(combine_feature, shape=[-1, c1])

        out = self.fc1(img_comb_feature)
        return out


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
            48 * 2, out_channels, weight_attr=weight_attr, bias_attr=bias_attr)
        self.fc2 = nn.Linear(
            96, out_channels, weight_attr=weight_attr, bias_attr=bias_attr)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        # add gsrm
        self.max_length = 50
        self.num_heads = 8
        self.num_encoder_TUs = 2
        self.num_decoder_TUs = 4
        self.hidden_dims = 512
        self.pvam = PVAM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            hidden_dims=self.hidden_dims)
        self.gsrm = GSRM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            num_decoder_tus=self.num_decoder_TUs,
            hidden_dims=self.hidden_dims)
        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0

        self.vsfd = VSFD(in_channels=in_channels, char_num=self.char_num)

    def forward(self, x, targets=None):
        others = targets[-4:]
        encoder_word_pos = others[0]
        gsrm_word_pos = others[1]

        #b,c,h,w = x.shape #(bs, 512, 1, 80)
        #feature_dim = h*w
        #
        ##gsrm_feature, word_out, gsrm_out 
        #gsrm_feature, gsrm_out = self.gsrm(
        #    x, paddle.to_tensor(gsrm_word_pos), paddle.to_tensor(gsrm_slf_attn_bias1),
        #    paddle.to_tensor(gsrm_slf_attn_bias2))

        # ctc FC
        # B, C, H, W = x.shape
        # assert H == 1
        # x_ = x.squeeze(axis=2)
        # x_.stop_gradient = True
        # x_ = x_.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        # print("x_shape:", x_.shape) # [64, 80, 512]
        # x_, _ = self.lstm(x_) #[64, 80, 96]
        # print("x_ feature:", x_.shape) 
        # ctc_predicts = self.fc(x_)
        # print("ctc_predicts:", ctc_predicts.shape)
        #print("ctc_predicts shape:", x_.shape)
        pvam_feature = self.pvam(x, encoder_word_pos,
                                 gsrm_word_pos)  # [1, 80, 512]
        x_, _ = self.lstm(pvam_feature)
        ctc_predicts = self.fc2(x_)

        if not self.training:
            #if False:
            # others = targets[-4:]
            # encoder_word_pos = others[0]
            # gsrm_word_pos = others[1]
            ctc_predict = F.softmax(ctc_predicts, axis=2)
            predicts = {'predict': ctc_predict}

        else:
            # others = targets[-4:]
            # gsrm_word_pos = others[1]
            gsrm_slf_attn_bias1 = others[2]
            gsrm_slf_attn_bias2 = others[3]
            # encoder_word_pos = others[0]

            #print("x shape:", x.shape) 
            b, c, h, w = x.shape  #(bs, 512, 1, 80)
            feature_dim = h * w

            #gsrm_feature, word_out, gsrm_out 
            # pvam_feature = self.pvam(x, encoder_word_pos, gsrm_word_pos)
            # ctc_predicts = self.fc2(pvam_feature)
            gsrm_feature, gsrm_out = self.gsrm(
                pvam_feature,
                paddle.to_tensor(gsrm_word_pos),
                paddle.to_tensor(gsrm_slf_attn_bias1),
                paddle.to_tensor(gsrm_slf_attn_bias2))

            #final_out = self.vsfd(pvam_feature, gsrm_feature)
            # if not self.training:
            #     final_out = F.softmax(final_out, axis=1)

            predicts = OrderedDict([
                ('predict', ctc_predicts),
                #('predict', final_out),
                #('word_out', word_out),
                ('gsrm_out', gsrm_out),
                #('vsfd_out', final_out)
            ])
        return predicts
