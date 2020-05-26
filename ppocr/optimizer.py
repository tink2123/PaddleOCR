#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter


def AdamLr(params):
    lr = params["base_lr"]
    beta1 = params['beta1']
    beta2 = params['beta2']
    global_step = _decay_step_counter()
    learning_rate = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    beta1_pow = beta1**global_step
    beta2_pow = beta2**global_step
    decayed_lr = lr * fluid.layers.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    fluid.layers.tensor.assign(
        input=decayed_lr, output=learning_rate)

    return learning_rate

def AdamDecay(params, parameter_list=None):
    """
    define optimizer function
    args:
        params(dict): the super parameters
        parameter_list (list): list of Variable names to update to minimize loss
    return:
    """
    base_lr = params['base_lr']
    beta1 = params['beta1']
    beta2 = params['beta2']
    optimizer = fluid.optimizer.Adam(
        learning_rate=base_lr,
        beta1=beta1,
        beta2=beta2,
        parameter_list=parameter_list)
    decay_lr = AdamLr(params)
    return optimizer, decay_lr
