import json
from pathlib import Path
ann_dir = '/home/disk0/hyq/JSPT/data/1262'
annotation_files = Path(ann_dir).rglob('*.json')

model_class_names = ["others_unsure", "squatting_unsure", "sitting_unsure", "standing_unsure", "lying_unsure", "others", "squatting", "sitting", "standing", "lying"]
model_class_ids = [1,2,3,4,5,6,7,8,9,10]


categorie_ids = { 
    'others_unsure': 1,
    'squatting_unsure': 2,
    'sitting_unsure': 3,
    'standing_unsure': 4,
    'lying_unsure': 5,
    'others': 6,
    'squatting': 7,
    'sitting': 8,
    'standing': 9,
    'lying': 10
}

categories = [
    {'id': 1, 'name': 'others_unsure'},
    {'id': 2, 'name': 'squatting_unsure'},
    {'id': 3, 'name': 'sitting_unsure'},
    {'id': 4, 'name': 'standing_unsure'},
    {'id': 5, 'name': 'lying_unsure'},
    {'id': 6, 'name': 'others'},
    {'id': 7, 'name': 'squatting'},
    {'id': 8, 'name': 'sitting'},
    {'id': 9, 'name': 'standing'},
    {'id': 10, 'name': 'lying'},
]

images = []
annotations = []

print('prepare dataset')
for index, jsonfile in enumerate(annotation_files):
    # set image info
    f = open(jsonfile)
    anns = json.load(f)
    # set annotations info
    for ann in anns:
        temp_images = {
            'file_name': jsonfile.stem + '.jpg',
            'height': ann['height'],
            'width': ann['width'],
            'id': index
        }
        temp_annotations = {
            'segmentation': ann['segmentation'],
            'keypoints': ann['keypoints'],
            'num_keypoints': ann['num_keypoints'],
            'area': ann['area'],
            'iscrowd': 1 if ann['iscrowd'] else 0,
            'image_id': index,
            'bbox': ann['bbox'],
            'category_id': categorie_ids[ann['box_name']],
            'id': ann['id']
        }

        images.append(temp_images)
        annotations.append(temp_annotations)

result_json = {
    'images': images,
    'annotations': annotations,
    'categories': categories
}
with open('./train.json', 'w') as f:
    json.dump(result_json, f)

print("done")


