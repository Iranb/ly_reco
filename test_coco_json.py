from xtcocotools.coco import COCO

coco = COCO('/project/train/src_repo/code/train.json')
print(coco.getImgIds())