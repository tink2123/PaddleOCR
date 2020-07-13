# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid

from ppocr.utils.utility import create_module
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from copy import deepcopy


class RecModel(object):
    def __init__(self, params):
        super(RecModel, self).__init__()
        global_params = params['Global']
        char_num = global_params['char_ops'].get_char_num()
        global_params['char_num'] = char_num
        if "TPS" in params:
            tps_params = deepcopy(params["TPS"])
            tps_params.update(global_params)
            self.tps = create_module(tps_params['function'])\
                (params=tps_params)
        else:
            self.tps = None

        backbone_params = deepcopy(params["Backbone"])
        backbone_params.update(global_params)
        self.backbone = create_module(backbone_params['function'])\
                (params=backbone_params)

        head_params = deepcopy(params["Head"])
        head_params.update(global_params)
        self.head = create_module(head_params['function'])\
                (params=head_params)

        loss_params = deepcopy(params["Loss"])
        loss_params.update(global_params)
        self.loss = create_module(loss_params['function'])\
                (params=loss_params)

        self.loss_type = global_params['loss_type']
        self.image_shape = global_params['image_shape']
        self.max_text_length = global_params['max_text_length']
        if "num_heads" in global_params:
            self.num_heads = global_params["num_heads"]

    def create_feed(self, mode):
        image_shape = deepcopy(self.image_shape)
        image_shape.insert(0, -1)
        image = fluid.data(name='image', shape=image_shape, dtype='float32')
        if mode == "train":
            if self.loss_type == "attention":
                label_in = fluid.data(
                    name='label_in',
                    shape=[None, 1],
                    dtype='int32',
                    lod_level=1)
                label_out = fluid.data(
                    name='label_out',
                    shape=[None, 1],
                    dtype='int32',
                    lod_level=1)
                feed_list = [image, label_in, label_out]
                others = {'label_in': label_in, 'label_out': label_out}
            elif self.loss_type == "srn":
                encoder_word_pos = fluid.data(
                    name="encoder_word_pos",
                    shape=[
                        -1, int((image_shape[-2] / 8) * (image_shape[-1] / 8)),
                        1
                    ],
                    dtype="int64")
                gsrm_word_pos = fluid.data(
                    name="gsrm_word_pos",
                    shape=[-1, self.max_text_length, 1],
                    dtype="int64")
                gsrm_slf_attn_bias1 = fluid.data(
                    name="gsrm_slf_attn_bias1",
                    shape=[
                        -1, self.num_heads, self.max_text_length,
                        self.max_text_length
                    ])
                gsrm_slf_attn_bias2 = fluid.data(
                    name="gsrm_slf_attn_bias2",
                    shape=[
                        -1, self.num_heads, self.max_text_length,
                        self.max_text_length
                    ])
                label = fluid.data(
                    name='label', shape=[-1, 1], dtype='int32', lod_level=1)
                feed_list = [
                    image, label, encoder_word_pos, gsrm_word_pos,
                    gsrm_slf_attn_bias1, gsrm_slf_attn_bias2
                ]
                others = {
                    'label': label,
                    'encoder_word_pos': encoder_word_pos,
                    'gsrm_word_pos': gsrm_word_pos,
                    'gsrm_slf_attn_bias1': gsrm_slf_attn_bias1,
                    'gsrm_slf_attn_bias2': gsrm_slf_attn_bias2
                }
            else:
                label = fluid.data(
                    name='label', shape=[None, 1], dtype='int32', lod_level=1)
                feed_list = [image, label]
                others = {'label': label}
            loader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list,
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        else:
            others = None
            loader = None
            if self.loss_type == "srn":
                encoder_word_pos = fluid.data(
                    name="encoder_word_pos",
                    shape=[
                        -1, int((image_shape[-2] / 8) * (image_shape[-1] / 8)),
                        1
                    ],
                    dtype="int64")
                gsrm_word_pos = fluid.data(
                    name="gsrm_word_pos",
                    shape=[-1, self.max_text_length, 1],
                    dtype="int64")
                gsrm_slf_attn_bias1 = fluid.data(
                    name="gsrm_slf_attn_bias1",
                    shape=[
                        -1, self.num_heads, self.max_text_length,
                        self.max_text_length
                    ])
                gsrm_slf_attn_bias2 = fluid.data(
                    name="gsrm_slf_attn_bias2",
                    shape=[
                        -1, self.num_heads, self.max_text_length,
                        self.max_text_length
                    ])
                feed_list = [
                    image, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                    gsrm_slf_attn_bias2
                ]
                others = {
                    'encoder_word_pos': encoder_word_pos,
                    'gsrm_word_pos': gsrm_word_pos,
                    'gsrm_slf_attn_bias1': gsrm_slf_attn_bias1,
                    'gsrm_slf_attn_bias2': gsrm_slf_attn_bias2
                }
        return image, others, loader

    def __call__(self, mode):
        image, others, loader = self.create_feed(mode)
        if self.tps is None:
            inputs = image
        else:
            inputs = self.tps(image)
        conv_feas = self.backbone(inputs)
        predicts = self.head(conv_feas, others, mode)
        decoded_out = predicts['decoded_out']
        if mode == "train":
            loss = self.loss(predicts, others)
            if self.loss_type == "attention":
                label = others['label_out']
            else:
                label = others['label']
            outputs = {'total_loss':loss, 'decoded_out':\
                decoded_out, 'label':label}
            return loader, outputs

        elif mode == "export":
            predict = predicts['predict']
            if self.loss_type != "srn":
                predict = fluid.layers.softmax(predict)
            return [image, {'decoded_out': decoded_out, 'predicts': predict}]
        else:
            return loader, {'decoded_out': decoded_out}
