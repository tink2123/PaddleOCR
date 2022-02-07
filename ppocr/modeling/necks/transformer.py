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

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from paddle import nn
import paddle
import numpy as np

class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q,k,v = self.to_qkv(x).chunk(3, axis=-1)
        b,p,n,w = q.shape
        q = paddle.reshape(x=q, shape=[0, 0, self.heads, w//self.heads, n])
        k = paddle.reshape(x=k, shape=[0, 0, self.heads, w//self.heads, n])
        v = paddle.reshape(x=v, shape=[0, 0, self.heads, w//self.heads, n])

        dots = paddle.matmul(q, k.transpose(perm=[0,1,2,4,3])) * self.scale

        attn = nn.functional.softmax(dots)
        out = paddle.matmul(attn, v)
        out = paddle.reshape(out, [b,p,n,w])
        return self.to_out(out)

class Encoder(nn.Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm0(self.attn(x)) + x
        x = self.norm1(self.ffn(x)) + x
        return x


class Transformer(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(
                Encoder(dim, heads, dim_head, mlp_dim, dropout=0.)
                )
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class MobileViTBlock(nn.Layer):
    def __init__(self,
                 dim,
                 hidden_dim,
                 depth,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 patch_size=(2, 2)):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # local representations
        self.conv1 = ConvNormAct(dim, dim, padding=1)
        self.conv2 = ConvNormAct(dim, hidden_dim, kernel_size=1)

        # global representations
        self.transformer = Transformer(embed_dim=hidden_dim,
                                       num_heads=num_heads,
                                       depth=depth,
                                       qkv_bias=qkv_bias,
                                       mlp_ratio=mlp_ratio,
                                       dropout=dropout,
                                       attention_dropout=attention_dropout,
                                       droppath=droppath)

        # fusion
        self.conv3 = ConvNormAct(hidden_dim, dim, kernel_size=1)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvNormAct(2 * dim, dim, padding=1)
    
    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.conv2(x)
        # [B, 96, 32, 32]

        B, C, H, W = x.shape
        x = x.reshape([B, C, H//self.patch_h, self.patch_w, W//self.patch_w, self.patch_w])
        # [4, 96, 16, 2, 16, 2]
        x = x.transpose([0, 1, 3, 5, 2, 4])
        # [4, 96, 2, 2, 16, 16]
        x = x.reshape([B, C, (self.patch_h * self.patch_w), -1]) #[B, C, ws**2, n_windows**2]
        x = x.transpose([0, 2, 3, 1]) #[B, ws**2, n_windows**2, C]
        # [4, 4, 256, 96]
        x = self.transformer(x)
        x = x.reshape([B, self.patch_h, self.patch_w, H//self.patch_h, W//self.patch_w, C])
        x = x.transpose([0, 5, 3, 1, 4, 2])
        x = x.reshape([B, C, H, W])

        x = self.conv3(x)
        x = paddle.concat((h, x), axis=1)
        x = self.conv4(x)
        return x


if __name__ == "__main__":
    import paddle
    attention = Transformer(80, 2, heads=8, dim_head=64, mlp_dim=128, dropout=0.)
    x = paddle.randn([1,128,1,80])
    print(attention(x).shape)
