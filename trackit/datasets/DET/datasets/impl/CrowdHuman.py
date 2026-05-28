import json
import os
from trackit.datasets.DET.constructor import DetectionDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def _parse_annotation_file(annotation_file_path):
    annotation_entries = []
    with open(annotation_file_path, 'r', newline='\n') as f:
        for line in f:
            annotation_entries.append(json.loads(line))
    return annotation_entries


def _construct_CrowdHuman(constructor: DetectionDatasetConstructor, image_path, annotation_path):
    annotation_entries = _parse_annotation_file(annotation_path)
    constructor.set_total_number_of_images(len(annotation_entries))
    constructor.set_category_id_name_map({0: 'head', 1: 'person'})
    constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)
    for annotation_entry in annotation_entries:
        image_id = annotation_entry['ID']
        with constructor.new_image() as image_constructor:
            image_constructor.set_path(os.path.join(image_path, f'{image_id}.jpg'))
            for object_anno in annotation_entry['gtboxes']:
                if object_anno['tag'] != 'person':
                    continue

                human_bbox = object_anno['vbox']
                human_bbox_fine = True
                human_bbox_occluded = None
                human_bbox_box_id = None
                if 'extra' in object_anno:
                    human_bbox_extra = object_anno['extra']
                    if 'ignore' in human_bbox_extra:
                        if human_bbox_extra['ignore'] > 0:
                            human_bbox_fine = False
                        if 'box_id' in human_bbox_extra:
                            human_bbox_box_id = human_bbox_extra['box_id']
                        if 'occ' in human_bbox_extra:
                            human_bbox_occluded = human_bbox_extra['occ']

                if human_bbox_fine:
                    with image_constructor.new_object() as object_constructor:
                        object_constructor.set_bounding_box(human_bbox)
                        object_constructor.set_category_id(1)
                        if human_bbox_occluded is not None:
                            object_constructor.set_attribute('occluded', human_bbox_occluded)
                        if human_bbox_box_id is not None:
                            object_constructor.set_attribute('box_id', human_bbox_box_id)

                head_bbox = object_anno['hbox']
                head_bbox_fine = True
                head_bbox_occluded = None
                if 'head_attr' in object_anno:
                    head_attr = object_anno['head_attr']
                    if 'unsure' in head_attr:
                        if head_attr['unsure'] > 0:
                            head_bbox_fine = False
                    if 'ignore' in head_attr:
                        if head_attr['ignore'] > 0:
                            head_bbox_fine = False
                    if 'occ' in head_attr:
                        head_bbox_occluded = head_attr['occ']
                if head_bbox_fine:
                    with image_constructor.new_object() as object_constructor:
                        object_constructor.set_bounding_box(head_bbox)
                        object_constructor.set_category_id(0)
                        if head_bbox_occluded is not None:
                            object_constructor.set_attribute('occluded', head_bbox_occluded)
                        if human_bbox_box_id is not None:
                            object_constructor.set_attribute('box_id', human_bbox_box_id)
                image_constructor.set_attribute('human_full_bbox', object_anno['hbox'])


def construct_CrowdHuman(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    dataset_split = seed.data_split

    images_path = os.path.join(root_path, 'Images')
    if 'train' in dataset_split:
        _construct_CrowdHuman(constructor, images_path, os.path.join(root_path, 'annotation_train.odgt'))
    if 'val' in dataset_split:
        _construct_CrowdHuman(constructor, images_path, os.path.join(root_path, 'annotation_val.odgt'))
