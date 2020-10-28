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
gradient_clip = 10


class PVAM(nn.Layer):
    def __init__(self, in_channels, params):
        super(PVAM,self).__init__()
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
        self.flatten0 = paddle.nn.Flatten(start_axis=0,stop_axis=2)
        self.fc0 = paddle.nn.Linear(in_features=in_channels,
                                    out_features=in_channels,
                                    )
        self.emb = paddle.nn.Embedding(num_embeddings=self.max_length,embedding_dim=in_channels)
        self.flatten1 = paddle.nn.Flatten(stop_axis=0, start_axis=3)
        self.fc1 = paddle.nn.Linear(in_features=in_channels, out_features=1, bias_attr=False)

    def forward(self, inputs, encoder_word_pos, gsrm_word_pos):
        b, c, h, w = inputs.shape
        print("inputs.shape:",inputs.shape)
        conv_features = paddle.reshape(inputs, shape=[-1, c, h*w])
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])
        # transformer encoder
        b, t, c = conv_features.shape
        #encoder_word_pos = others["encoder_word_pos"]
        #gsrm_word_pos = others["gsrm_word_pos"]

        enc_inputs = [conv_features, encoder_word_pos, None]
        print("conv_features:",conv_features.shape)
        print("encoder_word_pos:",encoder_word_pos.shape)
        word_features = self.wrap_encoder_for_feature(enc_inputs)
        fluid.clip.set_gradient_clip(
            fluid.clip.GradientClipByValue(gradient_clip))

        # pvam
        b, t, c = word_features.shape
        word_features = self.fc0(word_features)
        word_features = paddle.reshape(word_features, [-1, -1, t, c])
        word_features = paddle.expand(word_features, [1, self.max_length, 1, 1])
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature = paddle.reshape(word_pos_feature, [-1, self.max_length, 1, c])
        word_pos_feature = paddle.expand(word_pos_feature, [1, 1, t, 1])
        y = word_pos_feature + word_features
        y = F.tanh(y)
        attention_weight = self.fc1(y)
        attention_weight = paddle.reshape(attention_weight, shape=[-1, self.max_length, t])
        attention_weight = paddle.nn.Softmax(attention_weight,axis=-1)
        pvam_features = paddle.matmul(attention_weight, word_features) #[b, max_length, c]
        return pvam_features


if __name__ == "__main__":

    import paddle
    paddle.disable_static()
    encoder_word_pos_np = np.random.random((1, 256, 1)).astype('int64')
    gsrm_word_pos_np = np.random.random((1,25,1)).astype('int64')
    encoder_word_pos = paddle.to_tensor(encoder_word_pos_np)
    gsrm_word_pos = paddle.to_tensor(gsrm_word_pos_np)
    others = {"encoder_word_pos":encoder_word_pos, "gsrm_word_pos":gsrm_word_pos}
    params = {"char_num": 38,
              "max_text_length": 25,
              "num_heads": 1,
              "num_encoder_TUs": 2,
              "num_decoder_TUs": 4,
              "hidden_dims":512
              }
    pvam = PVAM(in_channels=512, params=params)
    data_np = np.random.random((1, 512, 8, 32)).astype('float32')
    data = paddle.to_tensor(data_np)
    output = pvam(data, encoder_word_pos, gsrm_word_pos)



