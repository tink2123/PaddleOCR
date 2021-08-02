# recommended paddle.__version__ == 2.0.0
nohup python3.7 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_resnet_stn_bilstm_att.yml &
