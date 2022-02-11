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

from ppocr.modeling.necks.mobilevit import MobileViTBlock, ConvNormAct

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
                 dims=64, # XS
                 depth=2,
                 hidden_dims=[96, 120, 144]):                                                                                      
        super(EncoderWithTransformer, self).__init__()
        self.depth = depth
        # [B, 64, 32, 32]
        self.mvit_block_1 = MobileViTBlock(in_channels, hidden_dims[0], depth=2)
        self.conv1x1 = ConvNormAct(in_channels//4, dims, kernel_size=1) 
        self.out_channels = dims

    def forward(self, x):
        # [1,512,1,80]
        print("befor vit:", x.shape)
        x = self.mvit_block_1(x)
        print("after vit:",x.shape)
        x = self.conv1x1(x)                     
        return x  


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
                    self.encoder_reshape.out_channels)                     
            else:                                                                                                   
                self.encoder = support_encoder_dict[encoder_type](                                                  
                    self.encoder_reshape.out_channels, hidden_dims)                                                 
                                                                                                                    
            self.out_channels = self.encoder.out_channels                                                           
            self.only_reshape = False 
        self.conv = nn.Conv2D(
            in_channels,
            in_channels//2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)                                                                       
                                                                                                                    
    def forward(self, x):                                                                                           
        # x = self.encoder_reshape(x)                                                                               
        # if not self.only_reshape:                                                                                 
        #     x = self.encoder(x)                                                             
        x = self.encoder(x)
        x = x.squeeze(axis=2)
        x = paddle.transpose(x, perm=[0,2,1])
        print("x.shape:", x.shape)                                                                                  
        return x   
