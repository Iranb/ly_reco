export PYTHONPATH=$PWD:$PYTHONPATH
CONFIG_FILE='/home/disk0/hyq/JSPT/ly_reco/configs/body/2d_kpt_sview_rgb_img/associative_embedding/custom/res50_coco_512x512.py'
GPU_NUM=4
# python  /project/train/src_repo/code/create_json.py  
python tools/train.py ${CONFIG_FILE} --seed 1048596  --work-dir /home/disk0/hyq/JSPT/temp
# ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --seed 1048596  --work-dir /project/train/models
# bash /project/train/src_repo/code/train.sh