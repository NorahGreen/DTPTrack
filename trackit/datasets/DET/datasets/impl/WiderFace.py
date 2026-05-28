import os
from trackit.datasets.DET.constructor import DetectionDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def _construct_WiderFace(constructor: DetectionDatasetConstructor, images_path, groundtruth_file):
    dataset_attributes = {}
    with open(groundtruth_file, 'r', newline='\n') as f:
        while True:
            image_path = f.readline()
            image_path = image_path.strip()
            if image_path == '0 0 0 0 0 0 0 0 0 0':
                continue
            if len(image_path) == 0:
                break
            number_of_image_annotations = f.readline()
            number_of_image_annotations = number_of_image_annotations.strip()
            number_of_image_annotations = int(number_of_image_annotations)
            assert image_path not in dataset_attributes
            image_annotations = []
            for index in range(number_of_image_annotations):
                annotation = f.readline()
                annotation = annotation.strip()
                if annotation == '0 0 0 0 0 0 0 0 0 0':
                    continue
                annotations = annotation.split(' ')
                assert len(annotations) == 10
                bounding_box = annotations[0: 4]
                bounding_box = [int(v) for v in bounding_box]
                blur = int(annotations[4])
                expression = int(annotations[5])
                illumination = int(annotations[6])
                invalid = int(annotations[7])
                occlusion = int(annotations[8])
                pose = int(annotations[9])
                image_annotations.append((bounding_box, blur, expression, illumination, invalid, occlusion, pose))
            if len(image_annotations) > 0:
                dataset_attributes[image_path] = image_annotations

    constructor.set_total_number_of_images(len(dataset_attributes))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)
    constructor.set_category_id_name_map({0: 'face'})

    for image_path, image_annotation in dataset_attributes.items():
        with constructor.new_image() as image_constructor:
            image_constructor.set_path(os.path.join(images_path, image_path))
            for bounding_box, blur, expression, illumination, invalid, occlusion, pose in image_annotation:
                with image_constructor.new_object() as object_constructor:
                    object_constructor.set_bounding_box(bounding_box, occlusion == 2)
                    object_constructor.set_category_id(0)


def construct_WiderFace(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split

    if 'train' in data_split:
        _construct_WiderFace(constructor, os.path.join(root_path, 'images'), os.path.join(root_path, 'wider_face_split', 'wider_face_train_bbx_gt.txt'))
    if 'val' in data_split:
        _construct_WiderFace(constructor, os.path.join(root_path, 'images'), os.path.join(root_path, 'wider_face_split', 'wider_face_val_bbx_gt.txt'))
