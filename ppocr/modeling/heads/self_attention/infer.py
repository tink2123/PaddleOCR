import argparse
import ast
import multiprocessing
import numpy as np
import os
import sys
import cmdparser
from functools import partial
import time
import paddle
import paddle.fluid as fluid
import logging
from model_check import check_cuda
import reader
from config import *
from desc import *
from model import fast_decode as fast_decoder


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
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
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


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


def prepare_batch_input(insts, data_input_names, src_pad_idx, bos_idx, n_head,
                        d_model, place):
    """
    Put all padded data needed by beam search decoder into a dict.
    """
    input, decoder_input, y, labels = insts

    b, c, h, w = input.shape
    src_conv_seq_len = int(w / 8)
    _, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [[0] * src_conv_seq_len for i in range(len(input))],
        src_pad_idx,
        n_head,
        is_target=False)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    # start tokens
    trg_word = np.asarray([[bos_idx]] * len(input), dtype="int64")
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, 1, 1]).astype("float32")
    trg_word = trg_word.reshape(-1, 1, 1)

    def to_lodtensor(data, place, lod=None):
        data_tensor = fluid.LoDTensor()
        data_tensor.set(data, place)
        if lod is not None:
            data_tensor.set_lod(lod)
        return data_tensor

    # beamsearch_op must use tensors with lod
    init_score = to_lodtensor(
        np.zeros_like(
            trg_word, dtype="float32").reshape(-1, 1),
        place, [range(trg_word.shape[0] + 1)] * 2)
    trg_word = to_lodtensor(trg_word, place, [range(trg_word.shape[0] + 1)] * 2)
    init_idx = np.asarray(range(len(input)), dtype="int32")

    data_input_dict = dict(
        zip(data_input_names, [
            input, src_pos, src_slf_attn_bias, trg_word, init_score, init_idx,
            trg_src_attn_bias
        ]))
    return data_input_dict


def prepare_feed_dict_list(data_buffer, count, place):
    """
    Prepare the list of feed dict for multi-devices.
    """
    # feed_dict_list = []
    if data_buffer is not None:  # use_py_reader == False
        data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields
        # data = next(data_generator)
        # for idx, data_buffer in enumerate(data):
        data_input_dict = prepare_batch_input(
            data_buffer, data_input_names, ModelHyperParams.eos_idx,
            ModelHyperParams.bos_idx, ModelHyperParams.n_head,
            ModelHyperParams.d_model, place)
    feed_dict_list = data_input_dict
    return feed_dict_list
    #     feed_dict_list.append(data_input_dict)
    # return feed_dict_list if len(feed_dict_list) == count else None


def py_reader_provider_wrapper(data_reader, place):
    """
    Data provider needed by fluid.layers.py_reader.
    """

    def py_reader_provider():
        data_input_names = encoder_data_input_fields + fast_decoder_data_input_fields
        for batch_id, data in enumerate(data_reader()):
            data_input_dict = prepare_batch_input(
                data, data_input_names, ModelHyperParams.eos_idx,
                ModelHyperParams.bos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model, place)
            yield [data_input_dict[item] for item in data_input_names]

    return py_reader_provider


def fast_infer(args, test_dg=None, pass_id=0):
    """
    Inference by beam search decoder based solely on Fluid operators.
    """
    infer_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(infer_program, startup_prog):
        with fluid.unique_name.guard():
            out_ids, out_scores, pyreader = fast_decoder(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                args.n_layer,
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
                InferTaskConfig.beam_size,
                InferTaskConfig.max_out_len,
                ModelHyperParams.bos_idx,
                ModelHyperParams.eos_idx,
                use_py_reader=args.use_py_reader)

    # This is used here to set dropout to the test mode.
    infer_program = infer_program.clone(for_test=True)

    # if args.use_mem_opt:
    #     fluid.memory_optimize(infer_program)

    if InferTaskConfig.use_gpu:
        check_cuda(InferTaskConfig.use_gpu)
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    #print()
    if os.path.exists(InferTaskConfig.model_path):
        logging.info("load infer.model form" + InferTaskConfig.model_path)
        fluid.io.load_vars(
            exe,
            InferTaskConfig.model_path,
            vars=[
                var for var in infer_program.list_vars()
                if isinstance(var, fluid.framework.Parameter)
            ])

    #exec_strategy = fluid.ExecutionStrategy()
    # For faster executor
    #exec_strategy.use_experimental_executor = True
    #exec_strategy.num_threads = 1
    #build_strategy = fluid.BuildStrategy()
    #     if args.use_parallel_exe:
    #         infer_exe = fluid.ParallelExecutor(
    #             use_cuda=TrainTaskConfig.use_gpu,
    #             main_program=infer_program,
    #             #build_strategy=build_strategy,
    #             #exec_strategy=exec_strategy
    # )

    # data reader settings for inference
    # args.train_file_pattern = args.test_file_pattern
    # args.use_token_batch = False
    # args.sort_type = reader.SortType.NONE
    # args.shuffle = False
    # args.shuffle_batch = False
    # test_data = prepare_data_generator(
    #     args,
    #     is_test=False,
    #     count=dev_count,
    #     pyreader=pyreader,
    #     py_reader_provider_wrapper=py_reader_provider_wrapper,
    #     place=place)
    if args.use_py_reader:
        pyreader.start()
        data_generator = None
    else:
        data_generator = test_dg
    trg_idx2word = test_dg.idx2char

    cnt = 0
    count = 0

    flog = open('log-%d.txt' % pass_id, 'w')
    start_time = time.time()
    for data_buffer in data_generator:
        _, _, _, labels = data_buffer

        try:
            feed_dict_list = prepare_feed_dict_list(data_buffer, dev_count,
                                                    place)
            logging.debug(feed_dict_list.keys())
            # if args.use_parallel_exe:
            #     seq_ids, seq_scores = infer_exe.run(
            #         fetch_list=[out_ids.name, out_scores.name],
            #         feed=feed_dict_list,
            #         return_numpy=False)
            # else:
            seq_ids, seq_scores = exe.run(
                program=infer_program,
                fetch_list=[out_ids.name, out_scores.name],
                feed=feed_dict_list if feed_dict_list is not None else None,
                return_numpy=False,
                use_program_cache=False)
            seq_ids_list, seq_scores_list = [seq_ids], [
                seq_scores
            ] if isinstance(seq_ids,
                            paddle.fluid.LoDTensor) else (seq_ids, seq_scores)
            _i = 0
            for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):

                # How to parse the results:
                #   Suppose the lod of seq_ids is:
                #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                #   then from lod[0]:
                #     there are 2 source sentences, beam width is 3.
                #   from lod[1]:
                #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                _i = 0
                for i in range(len(seq_ids.lod()[0]) -
                               1):  # for each source sentence
                    # print("true is :", labels[cnt])
                    ref = -10000000000
                    ret = ''
                    _label = labels[_i]
                    logging.debug('label[%d]:' % _i + _label)
                    start = seq_ids.lod()[0][i]
                    end = seq_ids.lod()[0][i + 1]
                    for j in range(end - start):  # for each candidate
                        sub_start = seq_ids.lod()[1][start + j]
                        sub_end = seq_ids.lod()[1][start + j + 1]
                        _res = "".join([
                            trg_idx2word[idx]
                            for idx in post_process_seq(
                                np.array(seq_ids)[sub_start:sub_end])
                        ])
                        _score = np.array(seq_scores)[sub_end - 1]
                        hyps[i].append(_res)
                        scores[i].append(_score)
                        if _score > ref:
                            ret = _res
                            ref = _score
                        logging.debug("raw idx is" + str(
                            np.squeeze(np.array(seq_ids)[sub_start:sub_end])))
                        logging.debug(' score:%f ' % _score + ' pred:' + _res)

                        if len(hyps[i]) >= InferTaskConfig.n_best:
                            break

                    if ret == _label:
                        cnt = cnt + 1
                    _i = _i + 1
                    count = count + 1
                    logging.debug(str(cnt) + " " + str(count))
        except (StopIteration, fluid.core.EOFException):
            # The data pass is over.
            if args.use_py_reader:
                pyreader.reset()
            break
    logging.info('epoch %d ,infer accuracy is %f speed is %f s per pic' %
                 (pass_id, 1.0 * cnt / (count + 1e-9),
                  (time.time() - start_time) / (count + 1e-9)))


if __name__ == "__main__":
    import data_generator as dg
    args = cmdparser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.info("Prepare train_dg and vali_dg, it May take some time")
    batchsize = args.batch_size
    vocab_fpath = args.trg_vocab_fpath
    train_imgshape = [int(num) for num in args.train_imgshape.split(',')]
    filelist = args.val_filelist
    vali_dg = dg.datagen(filelist, args.batch_size, vocab_fpath, train_imgshape,
                         args.imgs_dir)
    fast_infer(args, pass_id=-1, test_dg=vali_dg)
