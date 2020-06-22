from model import transformer
from paddle import fluid
from paddle.fluid import layers
import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().setLevel(logging.INFO)
import multiprocessing
from config import *
from train import train_loop, test_context
from train import get_device_num
from infer import fast_infer
from model_check import check_cuda
import data_generator as dg
import cmdparser
#dataset
#lr
#d_model
args = cmdparser.parse_args()

if not TrainTaskConfig.use_gpu:
    os.environ['CPU_NUM'] = '1'
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
            TrainTaskConfig.label_smooth_eps,
            ModelHyperParams.bos_idx,
            use_py_reader=False,
            is_test=False)

        lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
            ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
        with fluid.default_main_program()._lr_schedule_guard():
            learning_rate = lr_decay * TrainTaskConfig.learning_rate
        optimizer = fluid.optimizer.Adam(
            learning_rate=args.lr,
            beta1=TrainTaskConfig.beta1,
            beta2=TrainTaskConfig.beta2,
            epsilon=TrainTaskConfig.eps)
        optimizer.minimize(avg_cost)
        logging.info("local start_up:")

        # sgd = fluid.optimizer.SGD(learning_rate=0.001)
        # sgd.minimize(sum_cost)

logging.info("Prepare train_dg and vali_dg, it May take some time")
batchsize = args.batch_size
filelist = args.train_filelist
vocab_fpath = args.trg_vocab_fpath
train_imgshape = [int(num) for num in args.train_imgshape.split(',')]
train_dg = dg.datagen(filelist, batchsize, vocab_fpath, train_imgshape,
                      args.imgs_dir)
# batchsize = InferTaskConfig.batch_size
filelist = args.val_filelist
vali_dg = dg.datagen(filelist, 1, vocab_fpath, train_imgshape, args.imgs_dir)

logging.info("train_looop")
# for pass_id in range(1,TrainTaskConfig.pass_num):
#train a epoch
train_exe = train_loop(
    exe,
    train_prog,
    startup_prog,
    dev_count,
    sum_cost,
    avg_cost,
    token_num,
    predict,
    None,
    args=args,
    train_dg=train_dg,
    pass_id=0,
    vali_dg=vali_dg)
#
# fluid.io.save_params(
#     exe,
#     os.path.join(TrainTaskConfig.model_dir,
#                   "latest"+ ".infer.model"),
#     train_prog)#"pass_" + str(pass_id)
#
# fluid.io.save_persistables(
#     exe,
#     os.path.join(TrainTaskConfig.ckpt_dir,
#                  "latest.checkpoint"), train_prog)
#
#
# test_context(exe,train_exe,dev_count,vali_dg,args= args,pass_id=pass_id)

# fast_infer(args, vali_dg)

# if not args.enable_ce:
#     fluid.io.save_persistables(
#         exe,
#         os.path.join(TrainTaskConfig.ckpt_dir,
#                      "pass_" + str(pass_id) + ".checkpoint"),
#         train_prog)
