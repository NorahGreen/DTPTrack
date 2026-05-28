import os
import json
import numpy as np
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


_category_id_name_map = {0: 'toy', 1: 'child', 2: 'face', 3: 'person', 4: 'cup'}


def construct_PTB(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence))]
    sequences.sort()

    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}
    constructor.set_total_number_of_sequences(len(sequences))
    constructor.set_category_id_name_map(_category_id_name_map)
    for sequence in sequences:
        current_dir = os.path.join(root_path, sequence)

        bbox_groundtruth_file = os.path.join(current_dir, '{}.txt'.format(sequence))
        rgb_image_file_dir = os.path.join(current_dir, 'rgb')
        frame_description_file = os.path.join(current_dir, 'frames.json')

        with open(frame_description_file, 'rb') as fid:
            frame_information = json.load(fid)

        image_timestamps = frame_information['imageTimestamp']
        image_frame_ids = frame_information['imageFrameID']

        if sequence == 'bear_front':
            category = 'toy'
        elif sequence == 'child_no1':
            category = 'child'
        elif sequence == 'face_occ5':
            category = 'face'
        elif sequence == 'new_ex_occ4':
            category = 'person'
        elif sequence == 'zcup_move_1':
            category = 'cup'
        else:
            raise Exception

        bounding_boxes = np.loadtxt(bbox_groundtruth_file, dtype=np.float64, delimiter=',')
        bounding_boxes = bounding_boxes[:, 0: 4]

        with constructor.new_sequence(category_name_id_map[category]) as sequence_constructor:
            sequence_constructor.set_name(sequence)

            for index_of_frame, bounding_box in enumerate(bounding_boxes):
                with sequence_constructor.new_frame() as frame_constructor:
                    image_path = os.path.join(rgb_image_file_dir,
                                              'r-{}-{}.png'.format(image_timestamps[index_of_frame], image_frame_ids[index_of_frame]))
                    frame_constructor.set_path(os.path.join(rgb_image_file_dir, image_path))
                    if not np.any(np.isnan(bounding_box)):
                        frame_constructor.set_bounding_box(bounding_box.tolist())
