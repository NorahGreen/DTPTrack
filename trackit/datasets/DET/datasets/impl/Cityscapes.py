import os
import json
from ....common.meta.cityscapes import name2label, labels
from ....common.types.bounding_box import BoundingBoxFormat
from ...constructor import DetectionDatasetConstructor
from ..Cityscapes import CityscapesAnnotationOption


def construct_CityScapes(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    annotation_option = seed.annotation_option
    things_only = seed.things_only

    constructor.set_category_id_name_map({label.id: label.name for label in labels})

    annotation_paths = []

    if 'train' in data_split:
        if annotation_option == CityscapesAnnotationOption.PreferFine:
            annotation_paths.append(('gtFine', 'train'))
            annotation_paths.append(('gtCoarse', 'train_extra'))
        elif annotation_option == CityscapesAnnotationOption.FineOnly:
            annotation_paths.append(('gtFine', 'train'))
        elif annotation_option == CityscapesAnnotationOption.CoarseOnly:
            annotation_paths.append(('gtCoarse', 'train'))
            annotation_paths.append(('gtCoarse', 'train_extra'))
        else:
            raise Exception

    if 'val' in data_split:
        if annotation_option == CityscapesAnnotationOption.PreferFine:
            annotation_paths.append(('gtFine', 'val'))
        elif annotation_option == CityscapesAnnotationOption.FineOnly:
            annotation_paths.append(('gtFine', 'val'))
        elif annotation_option == CityscapesAnnotationOption.CoarseOnly:
            annotation_paths.append(('gtCoarse', 'val'))
        else:
            raise Exception

    annotation_file_paths = []
    image_file_paths = []

    for annotation_path in annotation_paths:
        cities = os.listdir(os.path.join(root_path, *annotation_path))
        for city in cities:
            files = os.listdir(os.path.join(root_path, *annotation_path, city))
            files = [file for file in files if file.endswith('.json')]
            for json_file in files:
                json_file_path = os.path.join(root_path, *annotation_path, city, json_file)

                file_name_parts = json_file[:-5].split('_')
                image_file_name = file_name_parts[0] + '_' + file_name_parts[1] + '_' + file_name_parts[
                    2] + '_leftImg8bit.png'

                if image_file_name == 'troisdorf_000000_000073_leftImg8bit.png':
                    continue
                image_file_path = os.path.join(root_path, 'leftImg8bit', annotation_path[1], city, image_file_name)
                annotation_file_paths.append(json_file_path)
                image_file_paths.append(image_file_path)

    constructor.set_total_number_of_images(len(image_file_paths))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for annotation_file_path, image_file_path in zip(annotation_file_paths, image_file_paths):
        with open(annotation_file_path, 'r') as fid:
            file_content = fid.read()

        json_objects = json.loads(file_content)
        img_width = json_objects['imgWidth']
        img_height = json_objects['imgHeight']

        bounding_boxes = []
        class_ids = []

        for object in json_objects['objects']:
            class_ = object['label']
            if things_only:
                if class_ not in name2label:
                    continue
                if not name2label[class_].hasInstances:
                    continue

            xmin = img_width
            ymin = img_height
            xmax = 0
            ymax = 0

            label = name2label[class_]

            for x, y in object['polygon']:
                if xmin > x:
                    xmin = x
                if ymin > y:
                    ymin = y
                if xmax < x:
                    xmax = x
                if ymax < y:
                    ymax = y

            bounding_box = [xmin, ymin, xmax, ymax]
            bounding_boxes.append(bounding_box)
            class_ids.append(label.id)

        if len(bounding_boxes) > 0:
            with constructor.new_image() as image_constructor:
                image_constructor.set_path(image_file_path)
                for bounding_box, class_ in zip(bounding_boxes, class_ids):
                    with image_constructor.new_object() as object_constructor:
                        object_constructor.set_bounding_box(bounding_box)
                        object_constructor.set_category_id(class_)
