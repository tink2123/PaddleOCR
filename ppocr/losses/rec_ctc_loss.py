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

class CosineEmbeddingLoss(nn.Layer):
    def __init__(self, margin=0.):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-12

    def forward(self, x1, x2, target):
        tmp1 = paddle.fluid.layers.reduce_sum(x1 * x2, dim=-1)
        tmp2 = paddle.norm(x1, axis=-1) + self.epsilon
        tmp3 = paddle.norm(x2, axis=-1) + self.epsilon
        similarity = tmp1 / (tmp2 * tmp3)
        one_list = paddle.full_like(target, fill_value=1)
        out = paddle.fluid.layers.reduce_mean(
            paddle.where(
                paddle.equal(target, one_list), 1. - similarity,
                paddle.maximum(
                    paddle.zeros_like(similarity), similarity - self.margin)))
        return out


class CTCLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.loss_sem = CosineEmbeddingLoss()

    def forward(self, predicts, batch):
        targets = batch[1].astype("int64")
        label_lengths = batch[2].astype('int64')
        sem_target = batch[3].astype('float32')
        sem_target.stop_gradient=True
        embedding_vectors = predicts['embedding_vectors']
        rec_pred = predicts['rec_pred']

        label_target = paddle.ones([embedding_vectors.shape[0]])
        sem_loss = paddle.sum(
            self.loss_sem(embedding_vectors, sem_target, label_target))

        predicts = rec_pred.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        loss = loss.mean()  # sum
        loss = loss + sem_loss * 0.1

        return {'loss': loss}
