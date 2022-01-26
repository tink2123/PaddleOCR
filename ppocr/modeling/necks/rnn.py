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
import numpy as np

from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr
from ppocr.modeling.heads.self_attention import WrapEncoderForFeature


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
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001, k=in_channels)
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='reduce_encoder_fea')

    def forward(self, x):
        x = self.fc(x)
        return x

class EncoderWithTransformer(nn.Layer):                                                                             
    def __init__(self,                                                                                              
                 in_channels,                                                                                       
                 num_heads,                                                                                         
                 num_encoder_TUs,                                                                                   
                 hidden_dims):                                                                                      
        super(EncoderWithTransformer, self).__init__()                                                              
                                                                                                                    
        self.num_heads = num_heads                                                                         
        self.num_encoder_TUs = num_encoder_TUs                                                                      
        self.hidden_dims = hidden_dims                                                                              
        # encoder                                                                                                   
        t = 256                                                                                                     
        self.wrap_encoder_for_feature = WrapEncoderForFeature(                                                      
            src_vocab_size=1,                                                                                       
            max_length=t,                                                                                           
            n_layer=self.num_encoder_TUs,                                                                           
            n_head=self.num_heads,                                                                                  
            d_key=int(self.hidden_dims / self.num_heads),                                                           
            d_value=int(self.hidden_dims / self.num_heads),                                                         
            d_model=self.hidden_dims,                                                                               
            d_inner_hid=512,                                                                           
            prepostprocess_dropout=0.1,                                                                             
            attention_dropout=0.1,                                                                                  
            relu_dropout=0.1,                                                                                       
            preprocess_cmd="n",                                                                                     
            postprocess_cmd="da",                                                                                   
            weight_sharing=True)                                                                                    
        self.fc0 = paddle.nn.Linear(
            in_features=self.hidden_dims,
            out_features=64)
        self.out_channels = 64
    def forward(self, inputs):                                                                                      
        b, c, h, w = inputs.shape                                                                                   
        conv_features = paddle.reshape(inputs, shape=[-1, c, h * w])                                                
        conv_features = paddle.transpose(conv_features, perm=[0, 2, 1])                                             
        # transformer encoder                                                                                       
        b, t, c = conv_features.shape                                                                               
        word_pos = paddle.reshape(paddle.arange(h*w), shape=[h*w, 1])                                               
        encoder_word_pos = paddle.expand(word_pos, shape=[b,h*w,1])                                                 
        encoder_word_pos.stop_gradient = True                                                                       
        enc_inputs = [conv_features, encoder_word_pos, None]                                                        
        word_features = self.wrap_encoder_for_feature(enc_inputs)
        word_features = self.fc0(word_features)                                            
        return word_features  


class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_dims=48, num_heads=8 ,num_encoder_TUs=2, **kwargs):        
        super(SequenceEncoder, self).__init__()                                                                     
        self.encoder_reshape = Im2Seq(in_channels)                                                                  
        self.out_channels = self.encoder_reshape.out_channels                                                       
        if encoder_type == 'reshape':                                                                               
            self.only_reshape = True                                                                                
        else:                                                                                                       
            support_encoder_dict = {                                                                                
                'reshape': Im2Seq,                                                                                  
                'fc': EncoderWithFC,                                                                                
                'rnn': EncoderWithRNN,                                                                              
                'transformer': EncoderWithTransformer                                                               
            }                                                                                                       
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(                                    
                encoder_type, support_encoder_dict.keys())                                                          
            if encoder_type == "transformer":                                                                       
                self.encoder = support_encoder_dict[encoder_type](                                                  
                    self.encoder_reshape.out_channels, num_heads, num_encoder_TUs, hidden_dims)                     
            else:                                                                                                   
                self.encoder = support_encoder_dict[encoder_type](                                                  
                    self.encoder_reshape.out_channels, hidden_dims)                                                 
                                                                                                                    
            self.out_channels = self.encoder.out_channels                                                           
            self.only_reshape = False 
        self.conv = nn.Conv2D(
            in_channels,
            in_channels//4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)                                                                       
                                                                                                                    
    def forward(self, x):                                                                                           
        # x = self.encoder_reshape(x)                                                                               
        # if not self.only_reshape:                                                                                 
        #     x = self.encoder(x)    
        x = self.conv(x)                                                                        
        x = self.encoder(x)                                                                                         
        return x   
