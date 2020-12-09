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
import paddle.fluid.framework as framework
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
        #word_features = self.flatten0(word_features)
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
        #word_out = F.softmax(word_out)
        word_ids = paddle.argmax(F.softmax(word_out), axis=1)
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
        #word1.stop_gradient = True
        #word2.stop_gradient = True

        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]

        gsrm_feature1, enc_input1 = self.wrap_encoder0(enc_inputs_1)
        #print("wrap_encoder:", gsrm_feature1.numpy())
        gsrm_feature2, enc_input2 = self.wrap_encoder1(enc_inputs_2)

        gsrm_feature2 = F.pad(gsrm_feature2, [0, 1],
                              value=0.,
                              data_format="NLC")
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        gsrm_features = gsrm_feature1 + gsrm_feature2

        gsrm_out = self.mul(gsrm_features)

        b, t, c = gsrm_out.shape
        gsrm_out = paddle.reshape(gsrm_out, [-1, c])
        #gsrm_out = F.softmax(x=gsrm_out)

        return gsrm_features, word_out, gsrm_out, gsrm_features
        #return word_out


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
        gradient = combine_feature_
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
        #out = F.softmax(out)
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

        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0
        print(self.gsrm.wrap_encoder0.prepare_decoder.emb0.weight)
        print(self.gsrm.wrap_encoder0.prepare_decoder.emb0.weight.name)

    def forward(self, inputs, others):
        encoder_word_pos = others["encoder_word_pos"]
        gsrm_word_pos = others["gsrm_word_pos"]
        gsrm_slf_attn_bias1 = others["gsrm_slf_attn_bias1"]
        gsrm_slf_attn_bias2 = others["gsrm_slf_attn_bias2"]

        pvam_feature = self.pvam(inputs, encoder_word_pos, gsrm_word_pos)
        #word_out= self.gsrm(
        #    pvam_feature, gsrm_word_pos, gsrm_slf_attn_bias1,
        #    gsrm_slf_attn_bias2)

        gsrm_feature, word_out, gsrm_out, gradient = self.gsrm(
            pvam_feature, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2)
        #self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0

        final_out = self.vsfd(pvam_feature, gsrm_feature)
        #final_out = gsrm_out

        _, decoded_out = paddle.topk(final_out, k=1)

        predicts = {
            'pvam_feature': pvam_feature,
            'predict': final_out,
            'decoded_out': decoded_out,
            'word_out': word_out,
            'gsrm_out': gsrm_out,
            'gradient': inputs
        }

        return predicts


class SRNLoss(nn.Layer):
    def __init__(self, params):
        super(SRNLoss, self).__init__()
        self.char_num = params['char_num']
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predicts, others):
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predicts['gsrm_out']
        label = others['label']
        #lbl_weight = others['lbl_weight']

        casted_label = paddle.cast(x=label, dtype='int64')
        label = paddle.reshape(label, shape=[-1, 1])
        cost_word = self.loss_func(word_predict, label=label)
        #label = paddle.reshape(label, shape=[-1])
        #label.stop_gradient = True
        #print("label:", label.numpy())
        #print("cost_word:", np.sum(cost_word.numpy()))

        #print("word predict:", np.argmax(word_predict.numpy(), axis=1))
        #print("gsrm predict:", np.argmax(gsrm_predict.numpy(), axis=1))
        cost_gsrm = self.loss_func(gsrm_predict, label=label)
        cost_vsfd = self.loss_func(predict, label=label)

        #sum_cost = cost_word + cost_vsfd * 2.0 + cost_gsrm * 0.15
        #sum_cost = paddle.sum([cost_word, cost_vsfd, cost_gsrm])

        sum_cost = cost_word + cost_vsfd + cost_gsrm
        #print({"cost_word,":cost_word.numpy(),"cost_vsfd,":cost_vsfd.numpy(), "cost_gsrm,":cost_gsrm.numpy()})
        #return [sum_cost, cost_vsfd, cost_word]
        #sum_cost = cost_gsrm
        return sum_cost


if __name__ == "__main__":

    import numpy as np
    import unittest
    from srn_head_st import SRNPredict
    import cv2
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
                #data_np = np.random.random((1, 1, 64, 256)).astype('float32')*255
                img = cv2.imread("../../../doc/imgs_words/en/word_1.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = img.shape[0], img.shape[1]
                new_w = int(math.ceil(64 * w / float(h)))
                resize_img = cv2.resize(img, (new_w, 64)).astype("float32")
                resize_img -= 0.5
                resize_img /= 0.5
                padding_im = np.zeros((1, 1, 64, 256), dtype=np.float32)
                padding_im[:, :, :, 0:new_w] = resize_img
                data_np = np.array(padding_im).astype('float32')
                encoder_word_pos_np = np.array(range(0, 256)).reshape(
                    (256, 1)).astype('int64')

                gsrm_word_pos_np = np.array(range(0, 25)).reshape(
                    (25, 1)).astype('int64')

                gsrm_attn_bias_data = np.ones((1, 25, 25)).astype('float32')
                gsrm_slf_attn_bias1_np = np.triu(
                    gsrm_attn_bias_data, 1).reshape(
                        [-1, 1, 25, 25]).astype('float32')
                gsrm_slf_attn_bias1_np = np.tile(gsrm_slf_attn_bias1_np,
                                                 [1, 8, 1, 1]) * [-1e9]
                gsrm_slf_attn_bias1_np = np.array(
                    gsrm_slf_attn_bias1_np).astype('float32')
                gsrm_slf_attn_bias2_np = np.tril(
                    gsrm_attn_bias_data, -1).reshape(
                        [-1, 1, 25, 25]).astype('float32')
                gsrm_slf_attn_bias2_np = np.tile(gsrm_slf_attn_bias2_np,
                                                 [1, 8, 1, 1]) * [-1e9]
                gsrm_slf_attn_bias2_np = np.array(
                    gsrm_slf_attn_bias2_np).astype('float32')

                label_np = np.array([
                    3, 5, 12, 3, 5, 1, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                    37, 37, 37, 37, 37, 37, 37, 37, 37
                ]).astype('int64')
                #print(label_np.shape)

                encoder_word_pos = paddle.to_tensor(encoder_word_pos_np)

                gsrm_word_pos = paddle.to_tensor(gsrm_word_pos_np)
                gsrm_slf_attn_bias1 = paddle.to_tensor(gsrm_slf_attn_bias1_np)
                gsrm_slf_attn_bias2 = paddle.to_tensor(gsrm_slf_attn_bias2_np)
                label = paddle.to_tensor(label_np)
                label = paddle.reshape(label, shape=[-1, 1])
                #print(label)

                others = {
                    "label": label,
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
                #for p in srn.parameters():
                #    print("dy param:{} {}".format(p.name, p.shape))

                restore = np.load("../../../train_data/dy_param.npz")
                state_dict = backbone.state_dict()
                #for k, v in state_dict.items():
                backbone.set_dict(restore, use_structured_name=False)
                srn.set_dict(restore, use_structured_name=False)

                data = paddle.to_tensor(data_np)
                sgd = fluid.optimizer.SGDOptimizer(
                    learning_rate=0.001,
                    parameter_list=srn.parameters() + backbone.parameters())
                for i in range(100):
                    feature, x = backbone(data)
                    #print("backbone:", np.sum(feature.numpy()))
                    predicts = srn(feature, others)
                    word_predict_dy = predicts['predict']
                    pvam_features = predicts['pvam_feature']
                    gsrm_out = predicts["gsrm_out"]

                    #print("pvam_features:", np.sum(pvam_features.numpy()))
                    #print("gsrm out:", np.sum(np.abs(gsrm_out.numpy())))
                    #print("predict shape:", word_predict_dy.shape)
                    #print("mean predict:", np.mean(word_predict_dy.numpy()))
                    print(
                        "pvam predict:",
                        np.argmax(
                            word_predict_dy.numpy(), axis=1))
                    print("gsrm predict:", np.argmax(gsrm_out.numpy(), axis=1))

                    Loss = SRNLoss(params)
                    loss = Loss(predicts, others)
                    loss.backward()
                    #print("forward:", np.sum(np.abs(x.numpy())))
                    print("gradient:", np.sum(np.abs(x.gradient())))
                    # print("forward2:",
                    #       np.sum(np.abs(predicts['gradient'].numpy())))
                    # print("gradient2:",
                    #       np.sum(np.abs(predicts['gradient'].gradient())))

                    sgd.minimize(loss)
                    srn.clear_gradients()
                    backbone.clear_gradients()
                    print("dy_loss:", np.sum(loss.numpy()))

                #print("dy_result:", np.sum(predicts['predict'].numpy()))

            with fluid.scope_guard(scope):
                paddle.enable_static()
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10
                np.random.seed(1333)
                label_np = np.array([
                    3, 5, 12, 3, 5, 1, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
                    37, 37, 37, 37, 37, 37, 37, 37, 37
                ]).astype('int64')

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
                    shape=[1, 25, 25],
                    dtype="float32")
                gsrm_slf_attn_bias2 = fluid.layers.data(
                    name="gsrm_slf_attn_bias2",
                    shape=[1, 25, 25],
                    dtype="float32")
                label_st = fluid.layers.data(
                    name="label", shape=[25], dtype="int64")

                others = {
                    "encoder_word_pos": encoder_word_pos,
                    "gsrm_word_pos": gsrm_word_pos,
                    "gsrm_slf_attn_bias1": gsrm_slf_attn_bias1,
                    "gsrm_slf_attn_bias2": gsrm_slf_attn_bias2,
                }
                resnet = ResNet(params)
                #for p in resnet.parameters():
                #    print(p.name)
                srn = SRNPredict(params)
                feature = resnet(xyz)
                print("feature:", feature)
                out = srn(feature, others)

                label_st = fluid.layers.reshape(label_st, shape=[25, 1])

                word_predict = out["word_out"]
                final_out = out["predict"]
                gsrm_out = out["gsrm_out"]
                print("final_out:", final_out)
                pvam_feature = out["pvam_feature"]
                #print(word_predict.shape)
                print("label shape:", label_st.shape)

                #casted_label = fluid.layers.cast(x=label_st, dtype='int64')
                cost_word = fluid.layers.cross_entropy(
                    input=fluid.layers.softmax(word_predict), label=label_st)
                cost_word = fluid.layers.reshape(
                    x=fluid.layers.reduce_sum(cost_word), shape=[1])

                cost_gsrm = fluid.layers.cross_entropy(
                    input=fluid.layers.softmax(gsrm_out), label=label_st)
                cost_gsrm = fluid.layers.reshape(
                    x=fluid.layers.reduce_sum(cost_gsrm), shape=[1])

                cost_vsfd = fluid.layers.cross_entropy(
                    input=fluid.layers.softmax(final_out), label=label_st)
                cost_vsfd = fluid.layers.reshape(
                    x=fluid.layers.reduce_sum(cost_vsfd), shape=[1])

                sum_cost = fluid.layers.sum([cost_gsrm, cost_word, cost_vsfd])

                #sum_cost = cost_gsrm

                sgd = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
                sgd.minimize(fluid.layers.reduce_sum(sum_cost))

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                main_program = framework.default_main_program()
                dy_dict = {}
                restore = np.load("../../../train_data/dy_param.npz")
                i = 0
                for k, v in restore.items():
                    dy_dict[i] = v
                    i += 1
                st_dict = {}
                j = 0
                for p in main_program.all_parameters():
                    st_dict[p.name] = dy_dict[j]
                    j += 1

                for var in main_program.list_vars():
                    try:
                        ten = paddle.fluid.global_scope().find_var(
                            var.name).get_tensor()
                    except:
                        continue
                    if var.name == "learning_rate_0":
                        continue
                    ten.set(st_dict[var.name], place)

                #fluid.load(main_program, "../../../train_data/best_accuracy.pdparams", exe)

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
                    "xyz",
                    "bn_conv1.output.1.tmp_3",
                    "res5c.add.output.5.tmp_1",
                    "conv2d_4.tmp_1",
                    "res4f.add.output.5.tmp_1",
                    "fc_64.tmp_1@GRAD",
                    #"tmp_31", "tmp_31@GRAD" # gsrm_feature
                    #"fc_14.tmp_1", "fc_14.tmp_1@GRAD" # word_out
                    #"tmp_31","tmp_31@GRAD"  # gsrm_features
                    #"conv2d_4.tmp_1", "conv2d_4.tmp_1@GRAD" # inputs for head
                    "bn_conv1.output.1.tmp_3",
                    "bn_conv1.output.1.tmp_3@GRAD"  # first conv
                ]
                #for param in fluid.default_startup_program().global_block().all_parameters():
                #    print("st param: {} {}".format(param.name, param.shape))
                for i in range(10):
                    ret = exe.run(
                        fluid.default_main_program(),
                        fetch_list=[
                            gsrm_out, feature, cost_word, word_predict, "label"
                        ] + param_list,
                        feed={
                            'xyz': data_np,
                            'label': label_np,
                            'encoder_word_pos': encoder_word_pos_np,
                            'gsrm_word_pos': gsrm_word_pos_np,
                            'gsrm_slf_attn_bias1': gsrm_slf_attn_bias1_np,
                            'gsrm_slf_attn_bias2': gsrm_slf_attn_bias2_np
                        })
                    #print("backbone:", ret[1])
                    #print("mean predict:", np.mean(ret[3]))
                    print("predict:", np.argmax(ret[0], axis=1))
                    #print("lable:", ret[4])
                    #print("st loss:", np.sum(ret[2]))

                    #print({"cost_word":np.sum(ret[2])})
                    #"cost_gsrm":np.sum(ret[3]), "cost_vsfd":np.sum(ret[4])})
                    print("predict:", np.sum(np.abs(ret[0])))

                    print("forword:", np.sum(np.abs(ret[-2])))
                    print("gradient:", np.sum(np.abs(ret[-1])))
                    #print("gradient2:", np.sum(np.abs(ret[-3])))

                #for item in ret[1:]:
                #    print("st tmp:", np.sum(item))
                #print("emb2 weights:", np.sum(ret[-2]))

    unittest.main()
