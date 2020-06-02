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
import random
import os


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


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz  # Tx.dot(Ty)

def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img

def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    #hsv[:,:,1] = hsv[:,:,1]*(1+delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img

def doudong(img):
    """
    doudong
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        #img[:w-s,:h-s,:] = (img[:w-s,:h-s,:] + img[s:,s:,:])/2
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img

def add_gasuss_noise(image, mean=0, var=0.1):
    """
    add_gasuss_noise
    mean : 均值
    var : 方差
    """
    #image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + 0.5 * noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out

def rad(x):
    """
    rad
    """
    return x * np.pi / 180

def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5
        # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = [dst1, dst2, dst3, dst4]
    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))  # todo , incase zero ERROR

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx],
                         [0, 1., dy],
                         [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0],
                         [0, 1., 0],
                         [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio


def get_warpAffine(config):
    """
    get_warpAffine
    """
    # w,h,shearx,sheary,shrink = config.w,config.h,config.shearx,config.sheary,config.shrink
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    '''
    org = np.array([[0, 0],
                    [w, h],
                    [0, h]], np.float32)

    dst_x = np.array([[0, 0],
                    [w, h],
                    [0.3*w , h]], np.float32)
    #Tx = cv2.getAffineTransform(org,dst_x)

    org = np.array([[0, 0],
                    [w, h],
                    [w, 0]], np.float32)
    dst_y = np.array([[0, 0],
                    [w, h],
                    [w , 0.3*h]], np.float32)
    #Ty = cv2.getAffineTransform(org,dst_y)
    '''
    return rz  # Tx.dot(Ty)


class Config:
    """
    Config
    """

    def __init__(self, ):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10  # 是旋转
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE  # if random.random()>0.5 else cv2.BORDER_REFLECT
        self.affine = True
        self.perspective = True
        self.reverse = True
        self.noise = True
        self.blur = True
        self.color = True
        self.dou = True
        # self.d_x = 0
        # self.d_y = 0
        self.shrink = 1  # - random.random()*0.3

    def make_(self, w, h):
        """
        make_
        """
        self.w = w
        self.h = h

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 30 * flag()
        self.angley = random.random() * 15 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()  # 是旋转
        self.fov = 42
        self.r = 0
        self.shearx = 0  # random.random()*0.3*flag()
        self.sheary = 0  # random.random()*0.05*flag()
        # self.shrink = 1 - random.random()*0.3
        self.borderMode = cv2.BORDER_REPLICATE  # if random.random()>0.2 else cv2.BORDER_TRANSPARENT
        self.w = w
        self.h = h
        ra = random.random()
        self.perspective = True  # True if ra > 0.75  else False
        self.affine = False  # True #if ra <= 0.5 else False
        self.reverse = True  # if random.random() >0.5 else False
        self.noise = True  # if random.random() >0.5 else False
        self.dou = False
        self.blur = True  # if random.random() >0.5 else False
        self.color = True  # True #if random.random() >0.5 else False


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    # hsv[:,:,1] = hsv[:,:,1]*(1+delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def doudong(img):
    """
    doudong
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        # img[:w-s,:h-s,:] = (img[:w-s,:h-s,:] + img[s:,s:,:])/2
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    add_gasuss_noise
    mean : 均值
    var : 方差
    """
    # image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + 0.5 * noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


# myConfig = config()
def warp(img,ang):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config()
    config.make(w, h, ang)
    new_img = img
    r1, c1 = 0, 0
    ratio = 1.0
    if config.perspective:
        warpR, (r1, c1), ratio = get_warpR(config)
        # config.d_x = c1
        # config.d_y = r1
        # config.shrink = ratio
        new_img = cv2.warpPerspective(new_img, warpR, (w, h), borderMode=config.borderMode)
    # print(img.dtype,warpT.dtype,type(w),type(h))
    # img.astype('float32')
    if config.affine:
        warpT = get_warpAffine(config)
        new_img = cv2.warpAffine(new_img, warpT, (w, h), borderMode=config.borderMode)
    if config.blur:
        new_img = blur(new_img)
    if config.color:
        new_img = cvtColor(new_img)
    if config.dou:
        new_img = doudong(new_img)
    if config.noise:
        new_img = add_gasuss_noise(new_img)
    if config.reverse:
        new_img = 255 - new_img
    return new_img

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
            else:
                assert False, "Unsupport loss_type %s in process_image"\
                    % loss_type
    return (norm_img)

