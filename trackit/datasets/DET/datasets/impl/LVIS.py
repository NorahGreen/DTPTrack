import os
import json
from trackit.datasets.DET.constructor import DetectionDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def _construct_LVIS(constructor: DetectionDatasetConstructor, coco_image_path, annotation_file):
    with open(annotation_file, 'r', newline='\n') as f:
        annotation = json.load(f)

    coco_url_prefix = 'http://images.cocodataset.org/'

    dataset_attributes = {}

    constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)
    constructor.set_category_id_name_map({category_info['id']: category_info['name'] for category_info in annotation['categories']})

    for image_attribute in annotation['images']:
        assert image_attribute['id'] not in dataset_attributes
        coco_url: str = image_attribute['coco_url']
        assert coco_url.startswith(coco_url_prefix)
        image_path = coco_url[len(coco_url_prefix):]
        dataset_attributes[image_attribute['id']] = ((image_attribute['width'], image_attribute['height']), image_path, [], [])

    constructor.set_total_number_of_images(len(dataset_attributes))

    for object_attribute in annotation['annotations']:
        image_attribute = dataset_attributes[object_attribute['image_id']]
        image_object_bboxes = image_attribute[2]
        image_object_category_ids = image_attribute[3]

        image_object_bboxes.append(object_attribute['bbox'])
        image_object_category_ids.append(object_attribute['category_id'])

    for image_attribute in dataset_attributes.values():
        with constructor.new_image() as image_constructor:
            image_constructor.set_path(os.path.join(coco_image_path, image_attribute[1]), image_attribute[0])
            for object_bbox, object_category_id in zip(image_attribute[2], image_attribute[3]):
                with image_constructor.new_object() as object_constructor:
                    object_constructor.set_bounding_box(object_bbox)
                    object_constructor.set_category_id(object_category_id)


def construct_LVIS(constructor: DetectionDatasetConstructor, seed):
    coco_path = seed.root_path
    lvis_annotation_file_path = seed.lvis_annotation_file_path
    data_split = seed.data_split

    if 'train' in data_split:
        _construct_LVIS(constructor, os.path.join(coco_path, 'images'), os.path.join(lvis_annotation_file_path, 'lvis_v1_train.json'))

    if 'val' in data_split:
        _construct_LVIS(constructor, os.path.join(coco_path, 'images'), os.path.join(lvis_annotation_file_path, 'lvis_v1_val.json'))
