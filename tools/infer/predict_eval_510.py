# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
import json
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, det_times = self.text_detector(img)
        print("det time:", det_times)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), det_times))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, rec_times = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), rec_times))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res, {"det_times": det_times, "rec_times": rec_times}


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def read_img_from_file(label_file,root_path):
    print("label_file:", label_file)
    f = open(label_file) 
    data= f.readlines()
    imgs_filename = []
    for idx, line in enumerate(data):
        img_name, _ = line.strip().split('\t')
        imgs_filename.append(root_path+img_name)
    assert len(imgs_filename) == len (data), "haha, bug raise"
    return imgs_filename


class Times(object):
    def __init__(self):
        self.det_times = {'det_preprocess':0, 'det_predict':0, 'det_postprocess':0, 'det_time': 0}
        self.rec_times = {'rec_preprocess':0, 'rec_predict':0, 'rec_postprocess':0, 'num_img': 0, 'rec_time': 0}
        self.num_img = 0

    def update(self, times, type='det'):
        pass

    def clean(self):
        self.__init__()

    def add(self, times, mode='det'):
        if mode=='det':
            for k in self.det_times.keys():
                self.det_times[k] += times[k]
            self.num_img += 1
        elif mode=='rec':
            for k in self.rec_times.keys():
                self.rec_times[k] += times[k]
        else:
            raise ValueError ("Times add error")         

    def info(self):
        logger.info("===> total number of det image is : {}".format(self.num_img))
        logger.info("===> det_times: {}".format(self.det_times))
        logger.info("===> rec_times: {}".format(self.rec_times))


import pynvml
import psutil
import GPUtil

def get_current_memory_mb(gpu_id=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    gpu_percent = 0
    if gpu_id is not None:
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[gpu_id].load
        gpu_percent = gpu_load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024./ 1024.
    return cpu_mem, gpu_mem, gpu_percent



def main(args):
    # image_file_list = get_image_file_list(args.image_dir)
    # image_file_list = read_imgs(args.image_dir)
    # root_dir = "/paddle/OCR_benchmark/test_set/"
    #root_dir = "/paddle/OCR_benchmark/PaddleOCR/510-Eng/" 
    root_dir = "./510-Eng/"
    image_file_list = read_img_from_file(args.image_dir, root_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score

    OCRTimes = Times()
    save_res = []
    import random
    random.shuffle(image_file_list)
    memory = 0
    gpu_memory = 0
    gpu_percent = 0
    for idx, image_file in enumerate(image_file_list):
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res, times = text_sys(img)
        elapse = time.time() - starttime
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))

        #OCRTimes.add(times['det_times'], 'det')
        #OCRTimes.add(times['rec_times'], 'rec')
        # OCRTimes.info()
        # logger.info(times['det_times'])
        # logger.info(times['rec_times'])
        logger.info("===> idx [{}/{}] imgfile: {}".format(idx, len(image_file_list), image_file))
        
        dt_num = len(dt_boxes)
        preds = []
        for dno in range(dt_num):
            text, score = rec_res[dno]
            #preds.append({"transcription": text, "points": np.array(dt_boxes[dno]).tolist()})
            if score >= drop_score:
                preds.append({"transcription": text, "points": np.array(dt_boxes[dno]).tolist()})
                text_str = "%s, %.3f" % (text, score)
                #print(text_str)
        save_res.append(image_file + '\t' + json.dumps(preds, ensure_ascii=False)+'\n')
        
        for text, score in rec_res:
            logger.info("{}, {:.3f}".format(text, score))
        
        cpu_m, gpu_m, gpu_p = get_current_memory_mb(0)
        memory += cpu_m
        gpu_memory += gpu_m
        gpu_percent += gpu_p

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results_510/"
            #draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))
    """
    logger.info("CPU memory use: {} MB, gpu memory use:{} MB, gpu util: {} %".format(round(memory/OCRTimes.num_img, 4), round(gpu_memory/OCRTimes.num_img, 4), round(gpu_percent/OCRTimes.num_img*100, 4)))
    OCRTimes.info()
   
    for k in OCRTimes.det_times.keys():
        logger.info("==> det {} , {}".format(k, OCRTimes.det_times[k]/OCRTimes.num_img))
    
    for k in OCRTimes.rec_times.keys():
        logger.info("==> rec {} , {}".format(k, OCRTimes.rec_times[k]/OCRTimes.num_img))
    """ 
    with open(args.save_res_path, 'w') as f:
        f.writelines(save_res)
        f.close()
    print("The predicted results saved in {}".format(args.save_res_path))



if __name__ == "__main__":
    main(utility.parse_args())
