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

import math
import cv2
import numpy as np


def get_bounding_box_rect(pos):
    left = min(pos[0])
    right = max(pos[0])
    top = min(pos[1])
    bottom = max(pos[1])
    return [left, top, right, bottom]


def resize_norm_img(img, image_shape):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def resize_norm_img_srn(img, image_shape):
    imgC, imgH, imgW = image_shape

    img_black = np.zeros((imgH, imgW))
    im_hei = img.shape[0]
    im_wid = img.shape[1]

    if im_wid <= im_hei * 1:
        img_new = cv2.resize(img, (imgH * 1, imgH))   
    elif im_wid <= im_hei * 2:
        img_new = cv2.resize(img, (imgH * 2, imgH))   
    elif im_wid <= im_hei * 3:
        img_new = cv2.resize(img, (imgH * 3, imgH))   
    else:
        img_new = cv2.resize(img, (imgW, imgH))   
    #img_new = cv2.resize(img, (imgW, imgH))

    #img_new = cv2.resize(img_new,self.train_imgshape[1:]) #reshape the size
    img_np = np.asarray(img_new)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_black[:, 0:img_np.shape[1]] = img_np
    img_black = img_black[:, :, np.newaxis]

    row,col,c = img_black.shape
    c = 1 
    
    return np.reshape(img_black,(c,row,col)).astype(np.float32)


def get_img_data(value):
    """get_img_data"""
    if not value:
        return None
    imgdata = np.frombuffer(value, dtype='uint8')
    if imgdata is None:
        return None
    imgori = cv2.imdecode(imgdata, 1)
    if imgori is None:
        return None
    return imgori

def process_image(img,
                  image_shape,
                  label=None,
                  char_ops=None,
                  loss_type=None,
                  max_text_length=None):
    norm_img = resize_norm_img(img, image_shape)
    norm_img = norm_img[np.newaxis, :]
    if label is not None:
        char_num = char_ops.get_char_num()
        text = char_ops.encode(label)
        if len(text) == 0 or len(text) > max_text_length:
            return None
        else:
            if loss_type == "ctc":
                text = text.reshape(-1, 1)
                return (norm_img, text)
            elif loss_type == "attention":
                beg_flag_idx = char_ops.get_beg_end_flag_idx("beg")
                end_flag_idx = char_ops.get_beg_end_flag_idx("end")
                beg_text = np.append(beg_flag_idx, text)
                end_text = np.append(text, end_flag_idx)
                beg_text = beg_text.reshape(-1, 1)
                end_text = end_text.reshape(-1, 1)
                return (norm_img, beg_text, end_text)
            ##elif loss_type == "srn":
            ##    text_padded = text + [[0]* (max_text_length - len(text)) for x in text]
            ##    text = text_padded.reshape(-1, 1)
            ##    return (norm_img, text)
            else:
                assert False, "Unsupport loss_type %s in process_image"\
                    % loss_type
    return (norm_img)



def srn_other_inputs(image_shape,
                     num_heads,
                     max_text_length):

    imgC, imgH, imgW = image_shape
    feature_dim = int((imgH / 8) * (imgW / 8))

    encoder_word_pos = np.array(range(0, feature_dim)).reshape((feature_dim, 1)).astype('int64')
    gsrm_word_pos = np.array(range(0, max_text_length)).reshape((max_text_length, 1)).astype('int64')

    lbl_weight = np.array([37] * max_text_length).reshape((-1,1)).astype('int64')

    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length)) 
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape([-1, 1, max_text_length, max_text_length])
    gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1, [1, num_heads, 1, 1]) * [-1e9] 

    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape([-1, 1, max_text_length, max_text_length])
    gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2, [1, num_heads, 1, 1]) * [-1e9] 

    encoder_word_pos = encoder_word_pos[np.newaxis, :]
    gsrm_word_pos = gsrm_word_pos[np.newaxis, :]
    #print (gsrm_slf_attn_bias1.shape)
    #print (gsrm_slf_attn_bias2.shape)
    #gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1[np.newaxis, :]
    #gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2[np.newaxis, :]

    return [lbl_weight, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2]

def process_image_srn(img,
                      image_shape,
                      num_heads,
                      max_text_length,
                      label=None,
                      char_ops=None,
                      loss_type=None):
    norm_img = resize_norm_img_srn(img, image_shape)
    norm_img = norm_img[np.newaxis, :]
    [lbl_weight, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
        srn_other_inputs(image_shape, num_heads, max_text_length)

    if label is not None:
        char_num = char_ops.get_char_num()
        text = char_ops.encode(label)
        if len(text) == 0 or len(text) > max_text_length:
            return None
        else:
            if loss_type == "srn":
                #print (text)
                text_padded = [37] * max_text_length
                for i in range(len(text)):
                    text_padded[i] = text[i]
                    lbl_weight[i] = [1.0]
                text_padded = np.array(text_padded)
                text = text_padded.reshape(-1, 1)
                #print(lbl_weight)
                return (norm_img, text,encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2,lbl_weight)
            else:
                assert False, "Unsupport loss_type %s in process_image"\
                    % loss_type
    return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2)
