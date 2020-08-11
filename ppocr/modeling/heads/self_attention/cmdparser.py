import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--train_filelist",
        type=str,
        required=True,
        help="The pattern to match training data files. by this we can specify the dataset")
    parser.add_argument(
        "--imgs_dir",
        type=str,
        default='imgs_dir/',
        help="imgs_dir")
    parser.add_argument(
        "--saved_dir",
        type=str,
        default='./output',
        help = 'saved_dir/trained_model/lastest.infer.model)')
    parser.add_argument(
        "--n_layer",
        type=int,
        default=2,
        help='the num of encoder and decoder layers')
    parser.add_argument(
          "--save_steps",
          type=int,
          default=10,
          help='the frequency of saving and infer')
    parser.add_argument(
        "--init_dir",
        type=str,
        default=None,
        help="init_model dir")
    parser.add_argument(
        "--val_filelist",
        type=str,
        required=True,
        help="The pattern to match validation data files. by this we can specify the dataset")

    parser.add_argument(
        "--train_imgshape",
        type=str,
        default='1,512,48',
        help = "channel, width, height === c,w,h"
    )
    parser.add_argument(
        "--use_parallel_exe",
        type=ast.literal_eval,
        default=False,
    )
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=200000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=lambda x: str(x.encode().decode("unicode-escape")),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. ")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--local',
        type=ast.literal_eval,
        default=True,
        help='Whether to run as local mode.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")
    parser.add_argument(
        '--update_method',
        choices=("pserver", "nccl2"),
        default="pserver",
        help='Update method.')
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")
    parser.add_argument(
        "--enable_ce",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")
    parser.add_argument(
        "--use_mem_opt",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to use memory optimization.")
    parser.add_argument(
        "--use_py_reader",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use py_reader.")
    parser.add_argument(
        "--fetch_steps",
        type=int,
        default=10,
        help="The frequency to fetch and print output.")
    parser.add_argument(
        "--lr",
        type = float,
        default=0.001,
        help="The Learning Rate for optimizer")
    args = parser.parse_args()
    # Append args related to dict
    # src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    # # print(src_dict)
    # trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    # dict_args = [
    #     "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
    #     str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
    #     "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
    #     str(src_dict[args.special_token[2]])
    # ]
    # merge_cfg_from_list(args.opts + dict_args,
    #                     [TrainTaskConfig, ModelHyperParams])
    return args
