添加coco预训练模型的方法：
先下载对应的权重文件CHECKPOINT_FILE,在训练脚本后加：
```python
--resume-from ${CHECKPOINT_FILE}
```

COCO数据集中的bbox格式: [x,y,width,height]，符合给定数据集中的数据格式。


# TODO:
1. ly_reco/custom_models/detectors/associative_embedding_custom.py
    - 设计 bbox_head和cls_head
    - 根据 forward_train 方法中的output 作为cls_head输入和bbox输入，设计loss
    - output： list： len(humans), each human keypoint :[BatchSize, 34, H, W] 
    - 在 forward_test 增加对应的网络结构和输出，并将输出结果整合到返回值中的result json中

    goal_1: len(humans) [BatchSize, 34, H, W] -> len(humans) [B, num_class] class result
    goal_2: len(humans) [BatchSize, 34, H, W] -> len(humans) [B, x,y,width,height] bbox result