添加coco预训练模型的方法：
先下载对应的权重文件CHECKPOINT_FILE,在训练脚本后加：
```python
--resume-from ${CHECKPOINT_FILE}
```
