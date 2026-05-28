import os
import csv
from collections import namedtuple
from trackit.datasets.DET.constructor import DetectionDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def construct_OpenImages(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split

    class_mids = []
    class_names = []

    for line in open(os.path.join(root_path, 'class-descriptions-boxable.csv'), 'r', encoding='utf-8'):
        line = line.strip()
        if len(line) == 0:
            continue
        words = line.split(',')
        assert len(words) == 2
        class_mids.append(words[0])
        class_names.append(words[1])

    mid_index_mapper = {mid: index for index, mid in enumerate(class_mids)}

    def _construct_sub_dataset(images_path: str, annotation_file_path: str):
        constructor.set_category_id_name_map({index: name for index, name in enumerate(class_names)})

        images = {}

        with open(annotation_file_path, 'r', encoding='utf-8') as fid:
            csv_reader = csv.reader(fid)
            headings = next(csv_reader)
            Row = namedtuple('Row', headings)

            last_row_image = None
            image_annos = []
            for r in csv_reader:
                row = Row(*r)
                image_name = row.ImageID
                if last_row_image != image_name:
                    if last_row_image is not None:
                        images[last_row_image] = image_annos
                        image_annos = []
                    last_row_image = image_name
                image_annos.append(row)

            if last_row_image is not None:
                images[last_row_image] = image_annos

        constructor.set_total_number_of_images(len(images))
        constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

        for image_name, image_annos in images.items():
            with constructor.new_image() as image_constructor:
                image_constructor.set_path(os.path.join(images_path, image_name + '.jpg'))
                image_size = image_constructor.get_image_size()
                for image_anno in image_annos:
                    object_category = mid_index_mapper[image_anno.LabelName]
                    bounding_box = [float(image_anno.XMin) * image_size[0], float(image_anno.XMax) * image_size[0],
                                    float(image_anno.YMin) * image_size[1], float(image_anno.YMax) * image_size[1]]
                    with image_constructor.new_object() as object_constructor:
                        object_constructor.set_category_id(object_category)
                        object_constructor.set_bounding_box(bounding_box)
                        object_constructor.merge_attributes({'IsOccluded': image_anno.IsOccluded, 'IsTruncated': image_anno.IsTruncated, 'IsGroupOf': image_anno.IsGroupOf,
                         'IsDepiction': image_anno.IsDepiction, 'IsInside': image_anno.IsInside})

    if 'train' in data_split:
        _construct_sub_dataset(os.path.join(root_path, 'train'), os.path.join(root_path, 'oidv6-train-annotations-bbox.csv'))
    if 'val' in data_split:
        _construct_sub_dataset(os.path.join(root_path, 'validation'), os.path.join(root_path, 'validation-annotations-bbox.csv'))
    if 'test' in data_split:
        _construct_sub_dataset(os.path.join(root_path, 'test'), os.path.join(root_path, 'test-annotations-bbox.csv'))
