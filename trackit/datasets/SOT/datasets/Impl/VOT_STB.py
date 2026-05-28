import math
import os

from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def construct_VOT_STB(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    with open(os.path.join(root_path, 'list.txt'), 'r') as f:
        sequence_list = f.readlines()
        sequence_list = [x.strip() for x in sequence_list]

    constructor.set_total_number_of_sequences(len(sequence_list))
    for index_of_sequence, sequence in enumerate(sequence_list):
        sequence_path = os.path.join(root_path, sequence)
        sequence_properties = parse_meta_data_file(os.path.join(sequence_path, 'sequence'))

        bounding_boxes = parse_bounding_boxes_file(os.path.join(sequence_path, 'groundtruth.txt'))
        bounding_box_len = len(bounding_boxes[0])
        if index_of_sequence == 0:
            if bounding_box_len == 4:
                constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)
            else:
                constructor.set_bounding_box_format(BoundingBoxFormat.Polygon)
        # occlusion_tag_file_path = os.path.join(sequence_path, 'occlusion.tag')
        # if os.path.exists(occlusion_tag_file_path):
        #     occlusion_flags = np.loadtxt(occlusion_tag_file_path, dtype=bool)
        # else:
        #     occlusion_flags = None

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_properties['name'])
            if 'fps' in sequence_properties:
                sequence_constructor.set_fps(float(sequence_properties['fps']))
            image_path = sequence_properties['channels.color'].replace('\\', os.path.sep)
            for frame_index, bounding_box in enumerate(bounding_boxes):
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, image_path % (frame_index + 1)))
                    if bounding_box is not None:
                        frame_constructor.set_bounding_box(bounding_box)


def parse_meta_data_file(file_path: str):
    properties = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            groups = line.split('=')
            assert len(groups) == 2
            groups[0] = groups[0].strip()
            groups[1] = groups[1].strip()
            assert groups[0] not in properties
            properties[groups[0]] = groups[1]
    return properties


def _parse_bounding_box_from_mask_annotation(string_line: str):
    groups = string_line.split(',')
    x, y, w, h = groups[:4]
    x, y, w, h = float(x), float(y), float(w), float(h)
    return x, y, w, h


def _parse_bounding_box_normal(bounding_box):
    bounding_box = [float(x) for x in bounding_box]
    if any(math.isnan(x) for x in bounding_box):
        return None
    return bounding_box


def _parse_bounding_box(string_line: str):
    groups = string_line.split(',')
    if len(groups) == 1:
        return None
    if len(groups) == 4:
        return _parse_bounding_box_normal(groups)
    if len(groups) % 2 == 0 and len(groups) > 4:
        return _parse_bounding_box_normal(groups)
    return None


def parse_bounding_boxes_file(file_path: str):
    bounding_boxes = []
    for line in open(file_path, 'r', encoding='utf-8', newline=''):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('m'):
            bounding_box = _parse_bounding_box_from_mask_annotation(line[1:])
        else:
            bounding_box = _parse_bounding_box(line)
        bounding_boxes.append(bounding_box)
    return bounding_boxes
