import sys
sys.path.append('/project/train/src_repo/code/mmpose/')
sys.path.append('/project/train/src_repo/code/poseclassifier/')

import os
import json
import time
import warnings
import numpy as np
import cv2

import torch
from argparse import ArgumentParser

from mmpose.apis import inference_bottom_up_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo
from models import PoseClassifier


device = 'cuda:0'

# -----------------------------------------------------------------------------------------------------------------------------------
# higherhrnet_w48 - infer time - 
pose_config = '/project/train/src_repo/code/mmpose/custom/configs/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth'

# mobilenetv2 - infer time - 
# pose_config = '/project/train/src_repo/code/mmpose/custom/configs/associative_embedding/coco/mobilenetv2_coco_512x512.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth'

# -----------------------------------------------------------------------------------------------------------------------------------
det_cat_id = 1
kpt_thr = 0.3
pose_nms_thr = 0.9

# in_dim = 51
# out_dim = 8
# cls_checkpoint = '/project/train/src_repo/temp/models/pose_classifier_In51_LR0.005-key/best_pose_classifier.pth'
# label2pose_dict = {0: 'lying', 1: 'lying_unsure', 2: 'others', 3: 'sitting', 4: 'sitting_unsure', 5: 'squatting', 6: 'standing', 7: 'standing_unsure'}
# drop_v = False

# in_dim = 51
# out_dim = 10
# cls_checkpoint = '/project/train/models/pose_classifier/best_pose_classifier.pth'
# label2pose_dict = {0: 'lying', 1: 'lying_unsure', 2: 'others', 3: 'others_unsure', 4: 'sitting', 5: 'sitting_unsure', 6: 'squatting', 7: 'squatting_unsure', 8: 'standing', 9: 'standing_unsure'}
# drop_v = False

in_dim = 34
out_dim = 10
cls_checkpoint = '/project/train/models/pose_classifier/best_pose_classifier.pth'
label2pose_dict = {0: 'lying', 1: 'lying_unsure', 2: 'others', 3: 'others_unsure', 4: 'sitting', 5: 'sitting_unsure', 6: 'squatting', 7: 'squatting_unsure', 8: 'standing', 9: 'standing_unsure'}
drop_v = True

# ------------------------------------------------------------------------------------------------------------------------------------
# 模型榜，需要检测的类别名称
model_class_names = ["lying",
                     "sitting",
                     "squatting",
                     "standing",
                     "others",
                     "lying_unsure",
                     "sitting_unsure",
                     "squatting_unsure",
                     "standing_unsure",
                     "others_unsure"]

# 实战榜，需要检测的类别名称
alert_class_names = ["lying", "sitting"]


# ------------------------------------------------------------------------------------------------------------------------------------
def init():
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    cls_model = PoseClassifier(in_dim, out_dim)
    cls_model.load_state_dict(torch.load(cls_checkpoint, map_location=torch.device(device)))
    cls_model = cls_model.to(device)

    return {'pose_model': pose_model, 'cls_model': cls_model}


def process_image(models, input_image, args=None):
    pose_model = models['pose_model']
    cls_model = models['cls_model']

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    
    image_name = input_image
    h, w, c = input_image.shape

    # pose_s_time = time.time()
    pose_results, returned_outputs = inference_bottom_up_pose_model(
        pose_model,
        image_name,
        dataset=dataset,
        dataset_info=dataset_info,
        pose_nms_thr=pose_nms_thr,
        return_heatmap=False,
        outputs=None)
    # print(f'pose infer time: {time.time() - pose_s_time}s')
    
    # print(pose_results)

    # # show the results
    # radius = 4
    # thickness = 1
    # out_file = '/project/train/src_repo/temp/vis_result.jpg'
    # vis_pose_result(
    #     pose_model,
    #     image_name,
    #     pose_results,
    #     radius=radius,
    #     thickness=thickness,
    #     dataset=dataset,
    #     dataset_info=dataset_info,
    #     kpt_score_thr=kpt_thr,
    #     show=False,
    #     out_file=out_file)

    # cls_s_time = time.time()
    target_info, objects = [], []
    for obj in pose_results:
        keypoints = obj['keypoints']
        x1, y1, x2, y2, conf = np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1]), 1
        x, y, width, height = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        xy_ratio = 0.05
        x_m = width * xy_ratio
        y_m = height * xy_ratio
        x1 = max(0, x1-x_m)
        y1 = max(0, y1-y_m)
        x2 = min(w, x2+x_m)
        y2 = min(h, y2+y_m)
        
        x, y, width, height = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        if width <= 0 or height <= 0:
            continue
        keypoints = obj['keypoints'].flatten()
        for i, k in enumerate(keypoints):  # set v
            if i % 3 == 2:
                keypoints[i] = 2 if keypoints[i] > kpt_thr else 1
            else:
                keypoints[i] = int(keypoints[i])

        keypoints_norm = keypoints.copy()
        for i, k in enumerate(keypoints_norm):  # rescale
            if i % 3 == 0:
                keypoints_norm[i] = (k - x) / width
            elif i % 3 == 1:
                keypoints_norm[i] = (k - y) / height
            if keypoints_norm[i] < 0:
                keypoints_norm[i] = 0
        if drop_v:
                keypoints_norm = [k for i, k in enumerate(keypoints_norm) if i % 3 < 2]

        keypoints_data = torch.tensor(keypoints_norm).unsqueeze(dim=0).to(device)
        logits = cls_model(keypoints_data)
        pre_labels = logits.argmax(dim=1)[0]
        name = label2pose_dict[int(pre_labels.cpu().data)]
        
        # for JSON serializable
        keypoints = [int(k) for k in keypoints]
        conf = 1

        # x, y, width, height, name, conf, kpts = obj  ###检测框用x,y,width,height表示, 也可以
        # keypoints = kpts['keypoints']

        '''
        keypoints是一个长度为17*3的列表，里面的内容示例如下
        keypoints = [x1,y1,v1, ......,x17,y17,v17]
        总共17个点，每个点有3个元素值x,y,v
        x,y表示点的坐标值,v是个标志位,v为0时表示这个关键点没有标注（这种情况下您可以忽略这个关键点的标注），
        v为1时表示这个关键点标注了但是不可见（被遮挡了），v为2时表示这个关键点标注了同时也可见
        17个点依次对应的人体骨骼关键点名称是:
        ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        开发者需要按照这个名称顺序来存储关键点信息到keypoints里, 不能乱序
        '''

        if name in model_class_names:
            obj = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': conf, 'name': name,
                   'keypoints': {'keypoints': keypoints, 'score': 1}}
            objects.append(obj)

        if name in alert_class_names:
            alert_obj = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': conf, 'name': name,
                         'keypoints': {'keypoints': keypoints, 'score': 1}}
            target_info.append(alert_obj)
    # print(f'cls infer time: {time.time() - cls_s_time}s')

    target_count = len(target_info)
    is_alert = True if target_count > 0 else False

    return json.dumps(
        {'algorithm_data': {'is_alert': is_alert, 'target_count': target_count, 'target_info': target_info},
         'model_data': {"objects": objects}})


if __name__ == '__main__':
    # Test API
    img = cv2.imread('/home/data/1262/ZDSliedown20220720_V1_train_building_1_000167.jpg')
    predictor = init()
    # print(img.shape)

    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print(f'Time: {(e - s)}s, FPS: {1/(e - s)}')

'''
ev_sdk输出json样例
{"algorithm_data": {"is_alert": True, "target_count": 1, "target_info": [
    {"x": 1805, "y": 886, "width": 468, "height": 595, "confidence": 0.7937217950820923, "name": "lying",
     "keypoints": {"keypoints": [2161.423828125, 990.58984375, 1.0, 2161.423828125, 981.29296875, 1.0, 2161.423828125,
                                 981.29296875, 1.0, 2093.238525390625, 967.34765625, 1.0, 2124.231689453125,
                                 985.94140625, 1.0, 2031.251708984375, 995.23828125, 1.0, 2068.443603515625,
                                 1069.61328125, 1.0, 2093.238525390625, 1074.26171875, 1.0, 2124.231689453125,
                                 1190.47265625, 1.0, 2161.423828125, 1088.20703125, 1.0, 2198.615966796875,
                                 1130.04296875, 1.0, 1944.47021484375, 1185.82421875, 1.0, 2012.6556396484375,
                                 1236.95703125, 1.0, 2124.231689453125, 1106.80078125, 0.0, 2186.218505859375,
                                 1195.12109375, 1.0, 2130.430419921875, 1232.30859375, 0.0, 2173.8212890625,
                                 1246.25390625, 1.0], "score": 0.7535077333450317}}]}, 
                  "model_data": {"objects": [
    {"x": 1805, "y": 886, "width": 468, "height": 595, "confidence": 0.7937217950820923, "name": "standing",
     "keypoints": {"keypoints": [2161.423828125, 990.58984375, 1.0, 2161.423828125, 981.29296875, 1.0, 2161.423828125,
                                 981.29296875, 1.0, 2093.238525390625, 967.34765625, 1.0, 2124.231689453125,
                                 985.94140625, 1.0, 2031.251708984375, 995.23828125, 1.0, 2068.443603515625,
                                 1069.61328125, 1.0, 2093.238525390625, 1074.26171875, 1.0, 2124.231689453125,
                                 1190.47265625, 1.0, 2161.423828125, 1088.20703125, 1.0, 2198.615966796875,
                                 1130.04296875, 1.0, 1944.47021484375, 1185.82421875, 1.0, 2012.6556396484375,
                                 1236.95703125, 1.0, 2124.231689453125, 1106.80078125, 0.0, 2186.218505859375,
                                 1195.12109375, 1.0, 2130.430419921875, 1232.30859375, 0.0, 2173.8212890625,
                                 1246.25390625, 1.0], "score": 0.7535077333450317}}]}}  #score不影响测试分数
'''

'''
(1). 目标检测框使用f1‑score作为指标
(2). 关键点检测使用Average Precision (AP): AP at OKS=.50:.05:.95作为测试指标
测试配置权重参数w, 最终精度得分 = w * 第(1)项的精度分 + (1 - w) * 第(2)项的精度分
'''
