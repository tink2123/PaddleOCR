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

from paddle import nn
import paddle

from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr


class Im2Seq(nn.Layer):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x


class EncoderWithRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.hidden_size = hidden_size
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

        self.embed_fc = nn.Linear(300, hidden_size)

    def get_initial_state(self, embed, tile_times=1):
        assert embed.shape[1] == 300
        state = self.embed_fc(embed)  # N * sDim
        state = state.unsqueeze(1)
        trans_state = paddle.transpose(state, perm=[1, 0, 2]) # (1,4,384)
        # print("trans state:", trans_state.shape)
        state = paddle.tile(trans_state, repeat_times=[tile_times, 1, 1])  # (4,4,384)
        # print("state shape:", state.shape)
        # trans_state = paddle.transpose(state, perm=[1, 0, 2])  # (4,4,384)
        # print("train state:", trans_state.shape)
        # print("hidden_size:", self.hidden_size)
        # state = paddle.reshape(trans_state, shape=[4, -1, self.hidden_size])
        return state

    def forward(self, x, embed=None):
        if self.training:
            # print("x shape:", x.shape)
            # print("x.shape[0]", x.shape[0])
            prev_h = self.get_initial_state(embed, 4)
            prev_c = paddle.zeros(shape=[4, x.shape[0], self.hidden_size])
            x, _ = self.lstm(inputs=x, initial_states=(prev_h, prev_c))
        else:
            x, _ = self.lstm(x)
        return x



class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001, k=in_channels, name='reduce_encoder_fea')
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='reduce_encoder_fea')

    def forward(self, x):
        x = self.fc(x)
        return x


class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x, targets=None):
        if self.training:
            label, length, emb = targets
            x = self.encoder_reshape(x)
            emb.stop_gradient=True
            if not self.only_reshape:
                x = self.encoder(x, emb)
        else:
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x, None)
        return x
