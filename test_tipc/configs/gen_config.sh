model_list="rec_mv3_none_none_ctc_v2.0 rec_r34_vd_none_none_ctc_v2.0 rec_mv3_none_bilstm_ctc_v2.0 rec_r34_vd_none_bilstm_ctc_v2.0 rec_mv3_tps_bilstm_ctc_v2.0 rec_r34_vd_tps_bilstm_ctc_v2.0 ch_ppocr_server_v2.0_rec "
#model_list="rec_resnet_stn_bilstm_att_v2.0 "
IFS=$' '
for model in ${model_list[*]}; do
    config_name=${model%%_v2.0*}.yml
    echo ${config_name}
    str='===========================train_params===========================
model_name:xxxxxx
python:python3.7
gpu_list:0|0,1
Global.use_gpu:True|True
Global.auto_cast:null
Global.epoch_num:lite_train_lite_infer=5|whole_train_whole_infer=100
Global.save_model_dir:./output/
Train.loader.batch_size_per_card:lite_train_lite_infer=128|whole_train_whole_infer=128
Global.pretrained_model:null
train_model_name:latest
train_infer_img_dir:./inference/rec_inference
null:null
##
trainer:norm_train
norm_train:tools/train.py -c test_tipc/configs/xxxxxx/rec_icdar15_train.yml -o
pact_train:null
fpgm_train:null
distill_train:null
null:null
null:null
##
===========================eval_params===========================
eval:tools/eval.py -c test_tipc/configs/xxxxxx/rec_icdar15_train.yml -o
null:null
##
===========================infer_params===========================
Global.save_inference_dir:./output/
Global.pretrained_model:
norm_export:tools/export_model.py -c test_tipc/configs/xxxxxx/rec_icdar15_train.yml -o
quant_export:null
fpgm_export:null
distill_export:null
export1:null
export2:null
##
infer_model:null
infer_export:tools/export_model.py -c test_tipc/configs/xxxxxx/rec_icdar15_train.yml -o
infer_quant:False
inference:tools/infer/predict_rec.py --rec_char_dict_path=./ppocr/utils/ic15_dict.txt --rec_image_shape="3,32,100"
--use_gpu:True|False
--enable_mkldnn:True|False
--cpu_threads:1|6
--rec_batch_num:1|6
--use_tensorrt:True|False
--precision:fp32|int8
--rec_model_dir:
--image_dir:./inference/rec_inference
--save_log_path:./test/output/
--benchmark:True
null:null'
    #echo ${str}
    mkdir -p ${model}
    #cp /workspace/tools/PaddleOCR/configs/rec/${config_name} ${model}/
    cd ${model} && echo ${str//xxxxxx/${model}} > train_infer_python.txt
    #mv ${config_name} rec_icdar15_train.yml
    cd ..
done

