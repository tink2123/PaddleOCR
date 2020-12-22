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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import yaml
import time
import shutil
import paddle
import paddle.distributed as dist
from tqdm import tqdm
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict
from ppocr.utils.logging import get_logger
from ppocr.data import build_dataloader
import numpy as np


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


global_config = AttrDict()

default_config = {'Global': {'debug': False, }}


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    merge_config(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    return global_config


def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                assert (sub_key in cur)
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not paddle.fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          vdl_writer=None):
    cal_metric_during_train = config['Global'].get('cal_metric_during_train',
                                                   False)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']

    global_step = 0
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        logger.info(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    #restore = np.load("./dy_param.npz")
    #model.set_dict(restore, use_structured_name=False)
    model.train()

    if 'start_epoch' in best_model_dict:
        start_epoch = best_model_dict['start_epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epoch_num):
        if epoch > 0:
            train_dataloader = build_dataloader(config, 'Train', device, logger)

        for idx, batch in enumerate(train_dataloader):
            if idx >= len(train_dataloader):
                break
            lr = optimizer.get_lr()
            t1 = time.time()
            images = batch[0]
            others = batch[-4:]
            #print(others)
            preds = model(images, others)
            #for p in model.parameters():
            #    print("dy_parameters:{} {}".format(p.name, p.shape))
            loss = loss_class(preds, batch)
            #print("loss:", loss)
            avg_loss = loss['loss']
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # logger and visualdl
            stats = {k: v.numpy().mean() for k, v in loss.items()}
            stats['lr'] = lr
            train_stats.update(stats)

            #cal_metric_during_train = False
            if cal_metric_during_train:  # onlt rec and cls need
                batch = [item.numpy() for item in batch]
                #print("post process class:", post_process_class)
                #print("preds:", preds)
                post_result = post_process_class(preds, batch[1])
                eval_class(post_result, batch)
                metirc = eval_class.get_metric()
                train_stats.update(metirc)

            t2 = time.time()
            train_batch_elapse = t2 - t1

            if vdl_writer is not None and dist.get_rank() == 0:
                for k, v in train_stats.get().items():
                    vdl_writer.add_scalar('TRAIN/{}'.format(k), v, global_step)
                vdl_writer.add_scalar('TRAIN/lr', lr, global_step)

            if dist.get_rank(
            ) == 0 and global_step > 0 and global_step % print_batch_step == 0:
                logs = train_stats.log()
                strs = 'epoch: [{}/{}], iter: {}, {}, time: {:.3f}'.format(
                    epoch, epoch_num, global_step, logs, train_batch_elapse)
                logger.info(strs)
            # eval
            if global_step > start_eval_step and \
                    (global_step - start_eval_step) % eval_batch_step == 0 and dist.get_rank() == 0:
                cur_metirc = eval(model, valid_dataloader, post_process_class,
                                  eval_class)
                cur_metirc_str = 'cur metirc, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metirc.items()]))
                logger.info(cur_metirc_str)

                # logger metric
                if vdl_writer is not None:
                    for k, v in cur_metirc.items():
                        if isinstance(v, (float, int)):
                            vdl_writer.add_scalar('EVAL/{}'.format(k),
                                                  cur_metirc[k], global_step)
                if cur_metirc[main_indicator] >= best_model_dict[
                        main_indicator]:
                    best_model_dict.update(cur_metirc)
                    best_model_dict['best_epoch'] = epoch
                    save_model(
                        model,
                        optimizer,
                        save_model_dir,
                        logger,
                        is_best=True,
                        prefix='best_accuracy',
                        best_model_dict=best_model_dict,
                        epoch=epoch)
                best_str = 'best metirc, {}'.format(', '.join([
                    '{}: {}'.format(k, v) for k, v in best_model_dict.items()
                ]))
                logger.info(best_str)
                # logger best metric
                if vdl_writer is not None:
                    vdl_writer.add_scalar('EVAL/best_{}'.format(main_indicator),
                                          best_model_dict[main_indicator],
                                          global_step)
            global_step += 1
        if dist.get_rank() == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                is_best=False,
                prefix='latest',
                best_model_dict=best_model_dict,
                epoch=epoch)
        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                is_best=False,
                prefix='iter_epoch_{}'.format(epoch),
                best_model_dict=best_model_dict,
                epoch=epoch)
    best_str = 'best metirc, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if dist.get_rank() == 0 and vdl_writer is not None:
        vdl_writer.close()
    return


def eval(model, valid_dataloader, post_process_class, eval_class):
    model.eval()
    with paddle.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(total=len(valid_dataloader), desc='eval model:')
        for idx, batch in enumerate(valid_dataloader):
            if idx >= len(valid_dataloader):
                break
            images = batch[0]
            others = batch[-4:]
            start = time.time()
            preds = model(images, others)

            batch = [item.numpy() for item in batch]
            # Obtain usable results from post-processing methods
            post_result = post_process_class(preds, batch[1])
            total_time += time.time() - start
            # Evaluate the results of the current batch
            eval_class(post_result, batch)
            pbar.update(1)
            total_frame += len(images)
        # Get final metirc，eg. acc or hmean
        metirc = eval_class.get_metric()

    pbar.close()
    model.train()
    metirc['fps'] = total_frame / total_time
    return metirc


def preprocess():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN'
    ]

    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1

    # save_config
    save_model_dir = config['Global']['save_model_dir']
    os.makedirs(save_model_dir, exist_ok=True)
    with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)

    logger = get_logger(
        name='root', log_file='{}/train.log'.format(save_model_dir))
    if config['Global']['use_visualdl']:
        from visualdl import LogWriter
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        os.makedirs(vdl_writer_path, exist_ok=True)
        vdl_writer = LogWriter(logdir=vdl_writer_path)
    else:
        vdl_writer = None
    print_dict(config, logger)
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, device, logger, vdl_writer
