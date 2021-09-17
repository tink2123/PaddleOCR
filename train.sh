# recommended paddle.__version__ == 2.0.0
#python3.7 -m paddle.distributed.launch --log_dir=./debug/ --gpus '4,5,6,7'  tools/train.py -c configs/rec/ch_ex/rec_chinese_lite_train_v2.0_exp.yml
python3.7 -m paddle.distributed.launch --log_dir=./debug/ --gpus '4,5,6,7'  tools/train.py -c configs/rec/ch_ex/rec_chinese_crnn_seed.yml -o Global.pretrained_model=output/26w_se_crnn/best_accuracy
