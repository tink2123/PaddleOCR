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

import paddle
from paddle import nn


class CTCLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.srn_loss_func = paddle.nn.loss.CrossEntropyLoss(reduction="sum")

    def forward(self, predicts, batch):
        ctc_predicts = predicts["predict"]
        word_predict = predicts["word_out"]
        gsrm_predict = predicts["gsrm_out"]
        predicts = ctc_predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        ctc_loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        ctc_loss = paddle.reshape(x=paddle.sum(ctc_loss), shape=[1])

        casted_label = paddle.cast(x=labels, dtype='int64')
        casted_label = paddle.reshape(x=casted_label, shape=[-1, 1])
        cost_word = self.srn_loss_func(word_predict, label=casted_label)
        cost_gsrm = self.srn_loss_func(gsrm_predict, label=casted_label)
        cost_word = paddle.reshape(x=paddle.sum(cost_word), shape=[1])
        cost_gsrm = paddle.reshape(x=paddle.sum(cost_gsrm), shape=[1])

        sum_cost = ctc_loss * 3.0 + cost_word*2.0 + cost_gsrm * 0.15

        return {'loss': sum_cost, 'ctc_loss': ctc_loss, 'cost_gsrm': cost_gsrm, 'cost_word':cost_word}
