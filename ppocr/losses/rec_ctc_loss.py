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
import numpy as np
from paddle import nn



def convert_pred_to_pseudo_parallel(text_index):
    """ convert text-index into text-label. """
    [unl_N, unl_len] = text_index.shape
    ignored_tokens = [0]
    batch_size = len(text_index)
    pad_list = np.zeros(shape=text_index.shape, dtype="int64")
    for batch_idx in range(batch_size):
        char_list = []
        for idx in range(len(text_index[batch_idx])):
            if text_index[batch_idx][idx] in ignored_tokens:
                continue
                # only for predict
            if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                    batch_idx][idx]:
                continue
            char_list.append(int(text_index[batch_idx][idx]))
        # char_list = paddle.to_tensor(char_list, dtype="int64")
        unl_length = len(char_list)
        # print(pseudo[batch_idx][:unl_length])
        pad_list[batch_idx][:unl_length] = char_list
    pseudo = paddle.to_tensor(pad_list)

    eos_index = pseudo.argmin(
        axis=-1)  # find smallest index of extend word(eos bos pad blank)
    # print("eos index:", eos_index)
    eos_index.stop_gradient=True
    eos_index[(eos_index == 0) & (text_index[:, 0] > 0)] = unl_len - 1
    unl_length = eos_index
    new_eos_index = eos_index.expand([unl_len, unl_N]).transpose([1, 0])
    indexs = paddle.arange(0, unl_len).expand([unl_N, unl_len])
    pad_mask = (indexs - new_eos_index) > 0
    non_pad_mask = (pad_mask == False)
    # print("non_pad_mask:", non_pad_mask)
    
    return pseudo, non_pad_mask, unl_length

class CTCLoss(nn.Layer):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor(
            [N] * B, dtype='int64', place=paddle.CPUPlace())
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = paddle.exp(-loss)
            weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
            weight = paddle.square(weight)
            loss = paddle.multiply(loss, weight)
        loss = loss.mean()
        return {'loss': loss}


class KLDivLoss(nn.Layer):

    def __init__(self, **kwargs):
        super(KLDivLoss, self).__init__()

        self.kldiv_criterion = paddle.nn.KLDivLoss(
            reduction='batchmean')

        self.lambda_cons=1
        
    def forward(self,
                unl_logit,
                unl_logit2,
                iteration,
                total_iter,
                l_local_feat=None,
                l_logit=None,
                l_text=None):
                
        loss_SemiSL = None
        l_da = None
        l_confident_ratio = 0

        # self._update_ema_variables(online, iteration, self.opt.ema_alpha)

        _, unl_len, nclass = unl_logit.shape

        predicts_text = paddle.nn.functional.log_softmax(unl_logit, axis=2)
        preds_index = predicts_text.argmax(axis=2)
        sequence_score = predicts_text.max(axis=2)
        pseudo, non_pad_mask, unl_length = convert_pred_to_pseudo_parallel(preds_index)

        sequence_score[non_pad_mask == False] = 0
        sample_prob = sequence_score.sum(axis=-1).exp()
        # todo: yxt sample_prob = sequence_score.mean()?

        # mask = sample_prob.ge(0.5)
        # all confident_ratio > 0 
        # todo: don't fix value
        confident_threshold = paddle.full_like(sample_prob, 0.5)
        mask = paddle.greater_equal(sample_prob, confident_threshold)
        confident_mask = mask.reshape([-1, 1]).tile([1, unl_len])
        final_mask = (non_pad_mask & confident_mask)
        confident_ratio = mask.cast("float32").mean()

        print("confident_ratio :", confident_ratio)

        if confident_ratio > 0:
            # unl_score, unl_index = unl_logit2.log_softmax(dim=-1).max(dim=-1)
            predicts_text_2 = paddle.nn.functional.log_softmax(unl_logit2, axis=2)
            unl_index = predicts_text_2.argmax(axis=2)
            unl_score = predicts_text_2.max(axis=2)

            unl_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(unl_index)

            unl_pred = unl_pred[:, 1:]
            unl_score[non_pad_mask == False] = 0
            unl_prob = unl_score.sum(axis=-1).exp()

            # unl_mask = unl_prob.ge(self.opt.confident_threshold)
            unl_mask = paddle.greater_equal(unl_prob, confident_threshold)

            l_log_softmax = paddle.nn.functional.log_softmax(l_logit, axis=-1)
            l_score = l_log_softmax.max(axis=-1)
            l_index = l_log_softmax.argmax(axis=-1)
            # l_score, l_index = l_logit.log_softmax(axis=-1).max(axis=-1)

            l_pred, non_pad_mask, _ = convert_pred_to_pseudo_parallel(l_index)
            l_pred = l_pred[:, 1:]
            l_score[non_pad_mask == False] = 0
            l_prob = l_score.sum(axis=-1).exp()

            # todo: don't fix value
            l_confident_threshold = paddle.full_like(sample_prob, 0.6)

            # l_mask = l_prob.ge(self.opt.l_confident_threshold)

            l_mask = paddle.greater_equal(l_prob, l_confident_threshold)

            l_confident_ratio = l_mask.cast("float32").mean()
            s_confident_ratio = unl_mask.cast("float32").mean()

            # uda
            # todo: don't fix value
            uda_softmax_temp = 0.4

            # unl_logit1 = (unl_logit/
            #               uda_softmax_temp).softmax(axis=-1)
            unl_logit1 = paddle.nn.functional.log_softmax((unl_logit/uda_softmax_temp), axis=-1)
            unl_logit2 = paddle.nn.functional.log_softmax(unl_logit2, axis=-1)
            loss_SemiSL = confident_ratio * self.kldiv_criterion(
                unl_logit2[final_mask], unl_logit1[final_mask])

            # loss_SemiSL = self.lambda_cons * loss_SemiSL

        return loss_SemiSL, confident_ratio, l_confident_ratio