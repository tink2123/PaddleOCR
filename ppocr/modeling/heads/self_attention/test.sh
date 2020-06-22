export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export FlAGS_cudnn_deterministic=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=./so/:$LD_LIBRARY_PATH
export HADOOP_HOME=/home/hadoop/apps/hadoop-2.7.5
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
source ~/yudeli/venv/bin/activate
python infer.py --trg_vocab_fpath './'  --train_filelist head100.list  --val_filelist test_img_eng_syn_norm_2k.list   --batch_size 1 --lr 0.0001 --imgs_dir /home/vis/lixuan/Textnet/data/recog_no_combine/eng_syn_norm/data/eng_syn_norm  --fetch_steps 1  --use_parallel_exe False --n_layer 2 > log_eval_2layers_no_beam.txt
