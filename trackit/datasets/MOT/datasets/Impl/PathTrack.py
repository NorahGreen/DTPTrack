from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat
import os
import numpy as np


def _simple_find_xml_value(xml_string: str, string_offset, prefix, postfix):
    begin_offset = xml_string.find(prefix, string_offset)
    begin_offset += len(prefix)
    end_offset = xml_string.find(postfix, begin_offset)
    assert begin_offset != end_offset
    return xml_string[begin_offset: end_offset], end_offset + len(postfix)


def _construct_PathTrack(constructor: MultipleObjectTrackingDatasetConstructor, path):
    sequences = os.listdir(path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(path, sequence))]
    sequences.sort()

    constructor.set_total_number_of_sequences(len(sequences))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)
    constructor.set_category_id_name_map({0: 'person'})

    for sequence in sequences:
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)

            sequence_path = os.path.join(path, sequence)

            with open(os.path.join(sequence_path, 'info.xml'), 'r', newline='\n') as f:
                xml_string = f.read()
                fps, offset = _simple_find_xml_value(xml_string, 0, '<fps name="fps">', '</fps>')
                fps = float(fps)
                sequence_constructor.set_fps(fps)
                scene_type, offset = _simple_find_xml_value(xml_string, offset, '<scene_type name="scene_type">', '</scene_type>')
                camera_movement, _ = _simple_find_xml_value(xml_string, offset, '<camera_movement name="camera_movement">',
                                                            '</camera_movement>')
                sequence_constructor.set_attribute('scene_type', scene_type)
                sequence_constructor.set_attribute('camera_movement', camera_movement)

            sequence_frames_path = os.path.join(sequence_path, 'img1')
            images = os.listdir(sequence_frames_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            assert images[0] == '000001.jpg'

            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_frames_path, image))

            gt_file_path = os.path.join(sequence_path, 'Base', 'Base.txt')
            gt_matrix = np.loadtxt(gt_file_path, dtype=np.int64)
            object_ids = set()
            for record in gt_matrix:
                if record[10].item() != 0:
                    continue
                object_id = record[1].item()
                if object_id not in object_ids:
                    with sequence_constructor.new_object(object_id) as object_constructor:
                        object_constructor.set_category_id(0)
                    object_ids.add(object_id)
                frame_id = record[0].item()
                with sequence_constructor.open_frame(frame_id - 1) as frame_constructor:
                    with frame_constructor.new_object(object_id) as object_constructor:
                        object_constructor.set_bounding_box(record[2: 6].tolist())


def construct_PathTrack(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split

    if 'train' in data_split:
        _construct_PathTrack(constructor, os.path.join(root_path, 'train'))
    if 'val' in data_split:
        _construct_PathTrack(constructor, os.path.join(root_path, 'test'))
