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
import numpy as np
import os
import random
import traceback
from paddle.io import Dataset
from .imaug import transform, create_operators
from paddle.fluid.dataloader.collate import default_collate_fn
import cv2


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines
    
    # def get_image_max_wh_ratio(self, img_idx_list):
    #     data_line = data_line.decode('utf-8')
    #     substr = data_line.strip("\n").split(self.delimiter)
    #     file_name = substr[0]
    #     label = substr[1]
    #     img_path = os.path.join(self.data_dir, file_name)
    #     return


    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:2]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None or data['polys'].shape[1]!=4:
                continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        # if len(idx_all) < 2:
        #     return None
        # idx = idx_all[0]
        # idx_list = idx_all[1]
        # batch_list = []
        # self.max_wh_ratio = 0
        # for i in idx_list:
        #     print("in batch:", i)
        #     file_idx = self.data_idx_order_list[i]
        #     data_line = self.data_lines[file_idx]
        #     data_line = data_line.decode('utf-8')
        #     substr = data_line.strip("\n").split(self.delimiter)
        #     file_name = substr[0]
        #     label = substr[1]
        #     img_path = os.path.join(self.data_dir, file_name)
        #     img = cv2.imread(img_path)
        #     h = img.shape[0]
        #     w = img.shape[1]
        #     ratio = w / float(h)
        #     self.max_wh_ratio = max(self.max_wh_ratio, ratio)
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            # data = {'img_path': img_path, 'label': label, 'wh_ratio': self.max_wh_ratio}
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = data
            # outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)


class BatchCompose(object):
    def __init__(self, config, mode, logger):
        self.mode = mode.lower()
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        self.ops = create_operators(dataset_config['transforms'], global_config)

    def __call__(self, data):
        batch_data = {}
        self.max_wh_ratio = 0
        wh_ratio = []
        for s_data in data:
            img_path = s_data["img_path"]
            if len(s_data['label']) > 25:
                continue
            img = cv2.imread(img_path)
            h = img.shape[0]
            w = img.shape[1]
            ratio = w / float(h)
            wh_ratio.append(ratio)
        self.mid_wh_ratio = np.median(wh_ratio)
        self.avg_wh_ratio = np.mean(wh_ratio)
        self.max_wh_ratio = np.max(wh_ratio)
        print("max:{}, avg:{}, med:{}".format(self.max_wh_ratio,self.avg_wh_ratio, self.mid_wh_ratio))
        print("self.max_wh_ratio:", self.max_wh_ratio)
        #self.max_wh_ratio = max(self.max_wh_ratio, ratio)
        # batch data, if user-define batch function needed
        # use user-defined here
        batch_data=[]
        error_id = 0
        for s_data in data:
            s_data["wh_ratio"] = self.max_wh_ratio
            try:
                output = transform(s_data, self.ops)
                #print("img shape:", s_data[0].shape)
            except:
                output = None
            if output is None:
                error_id += 1
                continue
            batch_data.append(output)
        print("max wh ratio:", self.max_wh_ratio)
        tmp_batch = random.sample(batch_data,error_id)
        
        batch_data = default_collate_fn(batch_data)

        return batch_data