import argparse
import ast
import copy
import logging
import multiprocessing
import os
import subprocess
if os.environ.get('FLAGS_eager_delete_tensor_gb', None) is None:
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0'

import paddle
from infer import fast_infer
import six
import sys
import time

import numpy as np
import paddle.fluid as fluid

from model_check import check_cuda
import reader
from config import *
from desc import *
from model import transformer, position_encoding_init
from model import fast_decode as fast_decoder
import dist_utils

import cmdparser

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))


def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1: return 1
    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().count('\n')
    return device_num


def append_nccl2_prepare(startup_prog, trainer_id, worker_endpoints,
                         current_endpoint):
    assert (trainer_id >= 0 and len(worker_endpoints) > 1 and
            current_endpoint in worker_endpoints)
    eps = copy.deepcopy(worker_endpoints)
    eps.remove(current_endpoint)
    nccl_id_var = startup_prog.global_block().create_var(
        name="NCCLID", persistable=True, type=fluid.core.VarDesc.VarType.RAW)
    startup_prog.global_block().append_op(
        type="gen_nccl_id",
        inputs={},
        outputs={"NCCLID": nccl_id_var},
        attrs={
            "endpoint": current_endpoint,
            "endpoint_list": eps,
            "trainer_id": trainer_id
        })
    return nccl_id_var


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False,
                   max_len=0):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []

    if max_len == 0:
        max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    #if is_label:  # label weight
    inst_weight = np.array(
        [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    #else:  # position data
    inst_pos = np.array([
        list(range(0, len(inst))) + [0] * (max_len - len(inst))
        for inst in insts
    ])
    return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data1 = np.triu(slf_attn_bias_data,
                                          1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data1 = np.tile(slf_attn_bias_data1,
                                          [1, n_head, 1, 1]) * [-1e9]

            slf_attn_bias_data2 = np.tril(
                slf_attn_bias_data, -1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data2 = np.tile(slf_attn_bias_data2,
                                          [1, n_head, 1, 1]) * [-1e9]

            return_list += [
                slf_attn_bias_data1.astype("float32"),
                slf_attn_bias_data2.astype("float32")
            ]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            #mask current pos
            #import pdb
            #pdb.set_trace()

            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, 1, max_len, 1])
            slf_attn_bias_data[:, 0, range(0, max_len), range(0,
                                                              max_len)] = -1e9
            slf_attn_bias_data = np.tile(slf_attn_bias_data, [1, n_head, 1, 1])
            return_list += [slf_attn_bias_data.astype("float32"), 0]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, data_input_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.

    insts=(input,decoder_input,y)
    """
    input, decoder_input, y, labels = insts
    #input [b,3,224,224],decoder_input[[1,3,4],[4,5,6,4,4],[6,7]]
    b, c, h, w = input.shape
    src_conv_seq_len = int(w / 8)
    _, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [[0] * src_conv_seq_len for i in range(len(input))],
        src_pad_idx,
        n_head,
        is_target=False)
    # src_word = input
    src_pos = src_pos.reshape(-1, src_max_len, 1)

    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        decoder_input, trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        y,
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_input_dict = dict(
        zip(data_input_names, [
            input, src_pos, src_slf_attn_bias, trg_word, trg_pos,
            trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
        ]))

    return data_input_dict, np.asarray([num_token], dtype="float32")


def prepare_data_generator(args,
                           is_test,
                           count,
                           pyreader,
                           py_reader_provider_wrapper,
                           place=None):
    """
    Data generator wrapper for DataReader. If use py_reader, set the data
    provider for py_reader
    """
    # NOTE: If num_trainers > 1, the shuffle_seed must be set, because
    # the order of batch data generated by reader
    # must be the same in the respective processes.
    shuffle_seed = 1 if num_trainers > 1 else None
    data_reader = reader.DataReader(
        fpattern=args.val_file_pattern if is_test else args.train_file_pattern,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else count),
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_seed=shuffle_seed,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False).batch_generator

    def stack(data_reader, count, clip_last=True):
        def __impl__():
            res = []
            for item in data_reader():
                res.append(item)
                if len(res) == count:
                    yield res
                    res = []
            if len(res) == count:
                yield res
            elif not clip_last:
                data = []
                for item in res:
                    data += item
                if len(data) > count:
                    inst_num_per_part = len(data) // count
                    yield [
                        data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                        for i in range(count)
                    ]

        return __impl__

    def split(data_reader, count):
        def __impl__():
            for item in data_reader():
                inst_num_per_part = len(item) // count
                for i in range(count):
                    yield item[inst_num_per_part * i:inst_num_per_part * (i + 1
                                                                          )]

        return __impl__

    if not args.use_token_batch:
        # to make data on each device have similar token number
        data_reader = split(data_reader, count)
    if args.use_py_reader:
        train_reader = py_reader_provider_wrapper(data_reader, place)
        if num_trainers > 1:
            assert shuffle_seed is not None
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)
        pyreader.decorate_tensor_provider(train_reader)
        data_reader = None
    else:  # Data generator for multi-devices
        data_reader = stack(data_reader, count)
    return data_reader


def prepare_feed_dict_list(data_buffer, init_flag, count):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    if data_buffer is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        # data = next(data_generator)
        # for idx, data_buffer in enumerate(data):
        data_input_dict, num_token = prepare_batch_input(
            data_buffer, data_input_names, ModelHyperParams.eos_idx,
            ModelHyperParams.eos_idx, ModelHyperParams.n_head,
            ModelHyperParams.d_model)
        # print(data_input_dict.keys)

    return data_input_dict
    #     feed_dict_list.append(data_input_dict)
    # if init_flag:
    #     for idx in range(count):
    #         pos_enc_tables = dict()
    #         for pos_enc_param_name in pos_enc_param_names:
    #             pos_enc_tables[pos_enc_param_name] = position_encoding_init(
    #                 ModelHyperParams.max_length + 1, ModelHyperParams.d_model)
    #         if len(feed_dict_list) <= idx:
    #             feed_dict_list.append(pos_enc_tables)
    #         else:
    #             feed_dict_list[idx] = dict(
    #                 list(pos_enc_tables.items()) + list(feed_dict_list[idx]
    #                                                     .items()))
    #
    # return feed_dict_list if len(feed_dict_list) == count else None


def py_reader_provider_wrapper(data_reader, place):
    """
    Data provider needed by fluid.layers.py_reader.
    """

    def py_reader_provider():
        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        for batch_id, data in enumerate(data_reader()):
            data_input_dict, num_token = prepare_batch_input(
                data, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            total_dict = dict(data_input_dict.items())
            yield [total_dict[item] for item in data_input_names]

    return py_reader_provider


def post_process_seq(seq,
                     bos_idx=ModelHyperParams.bos_idx,
                     eos_idx=ModelHyperParams.eos_idx,
                     output_bos=InferTaskConfig.output_bos,
                     output_eos=InferTaskConfig.output_eos):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def test_context(exe, train_exe, dev_count, vali_dg=None, args=None, pass_id=0):
    # Context to do validation.
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        test_prog.random_seed = 1000
        startup_prog.random_seed = 1000
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=True)
    test_prog = test_prog.clone(for_test=True)
    # test_data = prepare_data_generator(
    #     args,
    #     is_test=True,
    #     count=dev_count,
    #     pyreader=pyreader,
    #     py_reader_provider_wrapper=py_reader_provider_wrapper)

    exe.run(startup_prog)  # to init pyreader for testing
    if TrainTaskConfig.ckpt_path:
        fluid.io.load_persistables(
            exe, TrainTaskConfig.ckpt_path, main_program=test_prog)

    build_strategy = fluid.BuildStrategy()
    if args.use_parallel_exe:
        test_exe = fluid.ParallelExecutor(
            use_cuda=TrainTaskConfig.use_gpu,
            main_program=test_prog,
            build_strategy=build_strategy,
            share_vars_from=train_exe)

        # train_exe = fluid.ParallelExecutor(
        #     use_cuda=TrainTaskConfig.use_gpu,
        #     loss_name=avg_cost.name,
        #     main_program=train_prog,
        #     build_strategy=build_strategy,
        #     exec_strategy=exec_strategy,
        #     num_trainers=nccl2_num_trainers,
        #     trainer_id=nccl2_trainer_id)

    test_total_cost = 0
    test_total_token = 0

    if args.use_py_reader:
        pyreader.start()
        data_generator = None
    else:
        data_generator = vali_dg

    for data_buffer in data_generator:
        try:

            feed_dict_list = prepare_feed_dict_list(data_buffer, False,
                                                    dev_count)
            if args.use_parallel_exe:
                outs = test_exe.run(
                    fetch_list=[sum_cost.name, token_num.name, predict],
                    feed=feed_dict_list)
            else:
                outs = exe.run(
                    test_prog,
                    fetch_list=[sum_cost.name, token_num.name, predict],
                    feed=feed_dict_list)

        except (StopIteration, fluid.core.EOFException):
            # The current pass is over.
            if args.use_py_reader:
                pyreader.reset()
            break
        sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
        test_total_cost += sum_cost_val.sum()
        test_total_token += token_num_val.sum()
    test_avg_cost = test_total_cost / test_total_token
    test_ppl = np.exp([min(test_avg_cost, 100)])

    logging.info("epoch: %d, val avg loss: %f, val ppl: %f," %
                 (pass_id, test_avg_cost, test_ppl))

    return test_avg_cost, test_ppl


def train_loop(exe,
               train_prog,
               startup_prog,
               dev_count,
               sum_cost,
               avg_cost,
               token_num,
               predict,
               pyreader,
               nccl2_num_trainers=1,
               nccl2_trainer_id=0,
               args=None,
               train_dg=None,
               pass_id=0,
               vali_dg=None):
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        exe.run(startup_prog)  # to init pyreader for training
        logging.info("load checkpoint from {}".format(
            TrainTaskConfig.ckpt_path))
        fluid.io.load_persistables(
            exe, TrainTaskConfig.ckpt_path, main_program=train_prog)
    else:
        logging.info("init fluid.framework.default_startup_program")
        exe.run(startup_prog)

    logging.info("load init model")
    if args.init_dir:

        def if_exist(var):
            return os.path.exists(os.path.join(args.init_dir, var.name))

        fluid.io.load_vars(exe, args.init_dir, predicate=if_exist)
    logging.info("begin reader")
    # train_data = prepare_data_generator(
    #     args,
    #     is_test=False,
    #     count=dev_count,
    #     pyreader=pyreader,
    #     py_reader_provider_wrapper=py_reader_provider_wrapper)

    # # For faster executor
    #exec_strategy = fluid.ExecutionStrategy()
    # exec_strategy.num_iteration_per_drop_scope = int(args.fetch_steps)
    #build_strategy = fluid.BuildStrategy()
    # build_strategy.memory_optimize = False
    # build_strategy.enable_inplace = True

    sum_cost.persistable = True
    token_num.persistable = True
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    # build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
    # build_strategy.fuse_all_optimizer_ops = True
    #
    # if num_trainers > 1 and args.use_py_reader and TrainTaskConfig.use_gpu:
    #     dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
    #     exec_strategy.num_threads = 1

    logging.info("begin executor")
    if args.use_parallel_exe:
        train_exe = fluid.ParallelExecutor(
            use_cuda=TrainTaskConfig.use_gpu,
            loss_name=avg_cost.name,
            main_program=train_prog,
            #build_strategy=build_strategy,
            #exec_strategy=exec_strategy,
            #num_trainers=nccl2_num_trainers,
            #trainer_id=nccl2_trainer_id
        )

    # if args.val_file_pattern is not None:
    #     test = test_context(exe, train_exe, dev_count,vali_dg)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
        (1. - TrainTaskConfig.label_smooth_eps
         )) + TrainTaskConfig.label_smooth_eps *
                        np.log(TrainTaskConfig.label_smooth_eps / (
                            ModelHyperParams.trg_vocab_size - 1) + 1e-20))

    step_idx = 0
    init_flag = True
    logging.info("begin train")

    pass_start_time = time.time()

    if args.use_py_reader:
        pyreader.start()
        data_generator = None
    else:
        data_generator = train_dg

    batch_id = 0
    for pass_id in range(1, TrainTaskConfig.pass_num):
        for data_buffer in data_generator:
            try:
                feed_dict_list = prepare_feed_dict_list(data_buffer, init_flag,
                                                        dev_count)
                logging.debug("lbl_word is :" + str(feed_dict_list["lbl_word"]))
                if args.use_parallel_exe:
                    outs = train_exe.run(fetch_list=[
                        sum_cost.name, token_num.name, predict.name
                    ],
                                         feed=feed_dict_list)
                else:
                    outs = exe.run(
                        train_prog,
                        # if step_idx % args.fetch_steps == 0 else [],
                        fetch_list=[
                            sum_cost.name, token_num.name, predict.name
                        ],
                        feed=feed_dict_list)
                logging.debug(str(outs[:2]))
                logging.debug(str(np.argmax(outs[2], axis=1)))

                if step_idx % args.fetch_steps == 0:
                    sum_cost_val, token_num_val = np.array(outs[0]), np.array(
                        outs[1])
                    # sum the cost from multi-devices
                    total_sum_cost = sum_cost_val.sum()
                    total_token_num = token_num_val.sum()
                    total_avg_cost = total_sum_cost / total_token_num

                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.fetch_steps / (time.time() - avg_batch_time)))
                        avg_batch_time = time.time()
                batch_id += 1
                step_idx += 1

                init_flag = False

            except (StopIteration, fluid.core.EOFException):
                # The current pass is over.
                if args.use_py_reader:
                    pyreader.reset()
                break

        time_consumed = time.time() - pass_start_time
        # Validate and save the persistable.
        # if args.val_file_pattern is not None:
        #     # val_avg_cost, val_ppl = test()
        #     # logging.info(
        #     #     "epoch: %d, val avg loss: %f, val normalized loss: %f, val ppl: %f,"
        #     #     " consumed %fs" % (pass_id, val_avg_cost,
        #     #                        val_avg_cost - loss_normalizer, val_ppl,
        #     #                        time_consumed))
        #     print("")
        # else:
        logging.info("epoch: %d, consumed %fs" % (pass_id, time_consumed))
        if pass_id % args.save_steps == 0:
            fluid.io.save_params(exe,
                                 os.path.join(TrainTaskConfig.model_dir,
                                              "latest" + ".infer.model"),
                                 train_prog)  # "pass_" + str(pass_id)

            fluid.io.save_persistables(
                exe,
                os.path.join(TrainTaskConfig.ckpt_dir,
                             "pass_" + str(pass_id) + ".ckpt.model"),
                train_prog)

            fast_infer(args, vali_dg)

    return train_exe if args.use_parallel_exe else None

    # if args.enable_ce:  # For CE
    #     print("kpis\ttrain_cost_card%d\t%f" % (dev_count, total_avg_cost))
    #     if args.val_file_pattern is not None:
    #         print("kpis\ttest_cost_card%d\t%f" % (dev_count, val_avg_cost))
    #     print("kpis\ttrain_duration_card%d\t%f" % (dev_count, time_consumed))


def train(args):
    # priority: ENV > args > config
    is_local = os.getenv("PADDLE_IS_LOCAL", "1")
    if is_local == '0':
        args.local = False
    logging.info(args)

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")

    if training_role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        check_cuda(TrainTaskConfig.use_gpu)
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = fluid.CUDAPlace(gpu_id)
        dev_count = get_device_num()

    exe = fluid.Executor(place)

    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    if args.enable_ce:
        train_prog.random_seed = 1000
        startup_prog.random_seed = 1000

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                TrainTaskConfig.label_smooth_eps,
                ModelHyperParams.bos_idx,
                use_py_reader=args.use_py_reader,
                is_test=False)

            optimizer = None
            if args.sync:
                lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                    ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
                logging.info("before adam")

                with fluid.default_main_program()._lr_schedule_guard():
                    learning_rate = lr_decay * TrainTaskConfig.learning_rate

                optimizer = fluid.optimizer.Adam(
                    learning_rate=learning_rate,
                    beta1=TrainTaskConfig.beta1,
                    beta2=TrainTaskConfig.beta2,
                    epsilon=TrainTaskConfig.eps)
            else:
                optimizer = fluid.optimizer.SGD(0.003)
            optimizer.minimize(avg_cost)

    if args.local:
        logging.info("local start_up:")
        train_loop(exe, train_prog, startup_prog, dev_count, sum_cost, avg_cost,
                   token_num, predict, pyreader)
    else:
        if args.update_method == "nccl2":
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            port = os.getenv("PADDLE_PORT")
            worker_ips = os.getenv("PADDLE_TRAINERS")
            worker_endpoints = []
            for ip in worker_ips.split(","):
                worker_endpoints.append(':'.join([ip, port]))
            trainers_num = len(worker_endpoints)
            current_endpoint = os.getenv("POD_IP") + ":" + port
            if trainer_id == 0:
                logging.info("train_id == 0, sleep 60s")
                time.sleep(60)
            logging.info("trainers_num:{}".format(trainers_num))
            logging.info("worker_endpoints:{}".format(worker_endpoints))
            logging.info("current_endpoint:{}".format(current_endpoint))
            append_nccl2_prepare(startup_prog, trainer_id, worker_endpoints,
                                 current_endpoint)
            train_loop(exe, train_prog, startup_prog, dev_count, sum_cost,
                       avg_cost, token_num, predict, pyreader, trainers_num,
                       trainer_id)
            return

        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))

        logging.info("pserver_endpoints:{}".format(pserver_endpoints))
        logging.info("current_endpoint:{}".format(current_endpoint))
        logging.info("trainer_id:{}".format(trainer_id))
        logging.info("pserver_ips:{}".format(pserver_ips))
        logging.info("port:{}".format(port))

        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers,
            program=train_prog,
            startup_program=startup_prog)

        if training_role == "PSERVER":
            logging.info("distributed: pserver started")
            current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                "PADDLE_PORT")
            if not current_endpoint:
                logging.critical("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)

            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            logging.info("distributed: trainer started")
            trainer_prog = t.get_trainer_program()

            train_loop(exe, train_prog, startup_prog, dev_count, sum_cost,
                       avg_cost, token_num, predict, pyreader)
        else:
            logging.critical(
                "environment var TRAINER_ROLE should be TRAINER os PSERVER")
            exit(1)


if __name__ == "__main__":
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    args = cmdparser.parse_args()
    train(args)
