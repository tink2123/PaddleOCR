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
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import paddle.fluid as fluid
import numpy as np
from self_attention import WrapEncoderForFeature
from self_attention import WrapEncoder
from paddle.static import Program
from ppocr.modeling.backbones.rec_resnet_fpn import ResNetFPN
from ppocr.modeling.heads.rec_resnet_fpn_st import ResNet
gradient_clip = 10


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
        conv_features = paddle.reshape(inputs, shape=[-1, c, h * w])
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])
        # transformer encoder
        b, t, c = conv_features.shape

        enc_inputs = [conv_features, encoder_word_pos, None]
        word_features = self.wrap_encoder_for_feature(enc_inputs)

        # pvam
        b, t, c = word_features.shape
        word_features = self.flatten0(word_features)
        word_features = self.fc0(word_features)
        word_features_ = paddle.reshape(word_features, [-1, 1, t, c])
        word_features_ = paddle.tile(word_features_, [1, self.max_length, 1, 1])
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature_ = paddle.reshape(word_pos_feature,
                                           [-1, self.max_length, 1, c])
        word_pos_feature_ = paddle.tile(word_pos_feature_, [1, 1, t, 1])
        y = word_pos_feature_ + word_features_
        y = F.tanh(y)
        attention_weight = self.fc1(y)
        attention_weight = paddle.reshape(
            attention_weight, shape=[-1, self.max_length, t])
        attention_weight = F.softmax(attention_weight, axis=-1)
        pvam_features = paddle.matmul(attention_weight,
                                      word_features)  #[b, max_length, c]
        #print("dy pvam feature:", np.sum(pvam_features.numpy()))
        return pvam_features


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

    def forward(self, inputs, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2):
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

        word1 = paddle.cast(word_ids, "float32")
        word1 = F.pad(word1, [1, 0], value=1.0 * pad_idx, data_format="NLC")
        word1 = paddle.cast(word1, "int64")
        word1 = word1[:, :-1, :]
        word2 = word_ids

        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]

        gsrm_feature1 = self.wrap_encoder0(enc_inputs_1)
        #print("wrap_encoder:", gsrm_feature1.numpy())
        gsrm_feature2 = self.wrap_encoder1(enc_inputs_2)

        gsrm_feature2 = F.pad(gsrm_feature2, [0, 1],
                              value=0.,
                              data_format="NLC")
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        gsrm_features = gsrm_feature1 + gsrm_feature2

        gsrm_out = self.mul(gsrm_features)

        b, t, c = gsrm_out.shape
        gsrm_out = paddle.reshape(gsrm_out, [-1, c])
        gsrm_out = F.softmax(x=gsrm_out)

        return gsrm_features, word_out, gsrm_out


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
        combine_feature = paddle.concat([pvam_feature, gsrm_feature], axis=2)
        img_comb_feature = paddle.reshape(combine_feature, shape=[-1, c1 + c2])
        img_comb_feature_map = self.fc0(img_comb_feature)
        img_comb_feature_map = F.sigmoid(img_comb_feature_map)
        img_comb_feature_map = paddle.reshape(
            img_comb_feature_map, shape=[-1, t, c1])
        combine_feature = img_comb_feature_map * pvam_feature + (
            1.0 - img_comb_feature_map) * gsrm_feature
        img_comb_feature = paddle.reshape(combine_feature, shape=[-1, c1])

        out = self.fc1(img_comb_feature)
        out = F.softmax(out)
        return out


class SRN(nn.Layer):
    def __init__(self, in_channels, params):
        super(SRN, self).__init__()
        self.char_num = params['char_num']
        self.max_length = params['max_text_length']

        self.num_heads = params['num_heads']
        self.num_encoder_TUs = params['num_encoder_TUs']
        self.num_decoder_TUs = params['num_decoder_TUs']
        self.hidden_dims = params['hidden_dims']
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
        self.vsfd = VSFD(in_channels=in_channels)

        #self.gsrm.wrap_encoder0.prepare_decoder.emb0 = self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.pvam.wrap_encoder_for_feature.prepare_encoder.emb

    def forward(self, inputs, others):
        encoder_word_pos = others["encoder_word_pos"]
        gsrm_word_pos = others["gsrm_word_pos"]
        gsrm_slf_attn_bias1 = others["gsrm_slf_attn_bias1"]
        gsrm_slf_attn_bias2 = others["gsrm_slf_attn_bias2"]

        pvam_feature = self.pvam(inputs, encoder_word_pos, gsrm_word_pos)
        gsrm_feature, word_out, gsrm_out = self.gsrm(
            pvam_feature, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2)
        self.gsrm.wrap_encoder0.prepare_decoder.emb0 = self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.pvam.wrap_encoder_for_feature.prepare_encoder.emb

        final_out = self.vsfd(pvam_feature, gsrm_feature)

        _, decoded_out = paddle.topk(final_out, k=1)

        predicts = {
            'predict': final_out,
            'decoded_out': decoded_out,
            'word_out': word_out,
            'gsrm_out': gsrm_out
        }
        return predicts


if __name__ == "__main__":

    import numpy as np
    import unittest
    from srn_head_st import SRNPredict
    import os

    import paddle.fluid as fluid

    class TestDygraph(unittest.TestCase):
        def test(self):
            startup = fluid.Program()
            startup.random_seed = 111
            main = fluid.Program()
            main.random_seed = 111
            scope = fluid.core.Scope()

            place = fluid.CPUPlace()

            with fluid.dygraph.guard(place):
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10
                dy_param_init_value = {}
                np.random.seed(1333)
                #data_np = np.random.random((1, 512, 8, 32)).astype('float32')
                data_np = np.random.random((1, 1, 64, 256)).astype('float32')
                encoder_word_pos_np = np.random.random(
                    (1, 256, 1)).astype('int64')
                pvam_feature_np = np.random.random(
                    (1, 25, 512)).astype('float32')
                gsrm_word_pos_np = np.random.random((1, 25, 1)).astype('int64')
                gsrm_slf_attn_bias1_np = np.random.random(
                    (1, 8, 25, 25)).astype('float32')
                gsrm_slf_attn_bias2_np = np.random.random(
                    (1, 8, 25, 25)).astype('float32')

                encoder_word_pos = paddle.to_tensor(encoder_word_pos_np)
                pvam_features = paddle.to_tensor(pvam_feature_np)
                gsrm_word_pos = paddle.to_tensor(gsrm_word_pos_np)
                gsrm_slf_attn_bias1 = paddle.to_tensor(gsrm_slf_attn_bias1_np)
                gsrm_slf_attn_bias2 = paddle.to_tensor(gsrm_slf_attn_bias2_np)

                others = {
                    "encoder_word_pos": encoder_word_pos,
                    "gsrm_word_pos": gsrm_word_pos,
                    "gsrm_slf_attn_bias1": gsrm_slf_attn_bias1,
                    "gsrm_slf_attn_bias2": gsrm_slf_attn_bias2
                }
                params = {
                    "char_num": 38,
                    "max_text_length": 25,
                    "num_heads": 8,
                    "num_encoder_TUs": 2,
                    "num_decoder_TUs": 4,
                    "hidden_dims": 512
                }
                backbone = ResNetFPN(in_channels=1)
                for p in backbone.parameters():
                    pass
                    #print("dy param:{} {}".format(p.name, p.shape))
                srn = SRN(in_channels=512, params=params)
                for p in srn.parameters():
                    print("dy param:{} {}".format(p.name, p.shape))

                data = paddle.to_tensor(data_np)
                feature = backbone(data)
                print("backbone:", np.sum(feature.numpy()))
                predicts = srn(feature, others)

                print("dy_result:", np.sum(predicts['predict'].numpy()))

            with fluid.scope_guard(scope):
                paddle.enable_static()
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10
                np.random.seed(1333)

                #xyz = fluid.layers.data(
                #    name='xyz', shape=[512, 8, 32], dtype='float32')
                xyz = fluid.layers.data(
                    name='xyz', shape=[1, 64, 256], dtype='float32')
                encoder_word_pos = fluid.layers.data(
                    "encoder_word_pos", shape=[256, 1], dtype="int64")
                gsrm_word_pos = fluid.layers.data(
                    name="gsrm_word_pos", shape=[25, 1], dtype="int64")
                gsrm_slf_attn_bias1 = fluid.layers.data(
                    name="gsrm_slf_attn_bias1",
                    shape=[8, 25, 25],
                    dtype="float32")
                gsrm_slf_attn_bias2 = fluid.layers.data(
                    name="gsrm_slf_attn_bias2",
                    shape=[8, 25, 25],
                    dtype="float32")

                others = {
                    "encoder_word_pos": encoder_word_pos,
                    "gsrm_word_pos": gsrm_word_pos,
                    "gsrm_slf_attn_bias1": gsrm_slf_attn_bias1,
                    "gsrm_slf_attn_bias2": gsrm_slf_attn_bias2
                }
                resnet = ResNet(params)
                #for p in resnet.parameters():
                #    print(p.name)
                srn = SRNPredict(params)
                feature = resnet(xyz)
                print("feature:", feature)
                out = srn(feature, others)
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                #param_list = ["tmp_2", "tmp_4","tmp_8","tmp_11","tmp_14", "tmp_17", "tmp_21","tmp_24", "tmp_27", "tmp_30"]
                """
                param_list = ["layer_norm_7.tmp_2", "dropout_15.tmp_0", "tmp_10",
                              "layer_norm_8.tmp_2", "dropout_17.tmp_0", "tmp_13",
                              "layer_norm_10.tmp_2", "dropout_21.tmp_0", "tmp_14",
                              "layer_norm_11.tmp_2", "dropout_23.tmp_0", "tmp_16",
                              "layer_norm_12.tmp_2", "dropout_25.tmp_0", "tmp_17",
                              "layer_norm_13.tmp_2", "layer_norm_14.tmp_2", "dropout_28.tmp_0", "tmp_20"
                              ]
                """
                param_list = [
                    "xyz", "bn_conv1.output.1.tmp_3",
                    "res5c.add.output.5.tmp_1", "conv2d_4.tmp_1",
                    "res4f.add.output.5.tmp_1", "res3d.add.output.5.tmp_1"
                ]
                #for param in fluid.default_startup_program().global_block().all_parameters():
                #    print("st param: {} {}".format(param.name, param.shape))
                ret = exe.run(fetch_list=[out['predict'], feature] + param_list,
                              feed={
                                  'xyz': data_np,
                                  'encoder_word_pos': encoder_word_pos_np,
                                  'gsrm_word_pos': gsrm_word_pos_np,
                                  'gsrm_slf_attn_bias1': gsrm_slf_attn_bias1_np,
                                  'gsrm_slf_attn_bias2': gsrm_slf_attn_bias2_np
                              })
                print("st_result:", np.sum(ret[0]))
                for item in ret[1:]:
                    print("st tmp:", np.sum(item))
                #print("emb2 weights:", np.sum(ret[-2]))

    unittest.main()
