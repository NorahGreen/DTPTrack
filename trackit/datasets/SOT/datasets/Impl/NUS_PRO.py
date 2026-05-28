import os
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat
import numpy as np


_category_id_name_map = {0: 'airplane', 1: 'boat', 2: 'car', 3: 'face', 4: 'helicopter', 5: 'motorcycle person', 6: 'person'}


def construct_NUSPRO(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence))]
    sequences.sort()
    constructor.set_category_id_name_map(_category_id_name_map)
    constructor.set_total_number_of_sequences(len(sequences))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}
    for sequence in sequences:
        sequence_path = os.path.join(root_path, sequence)
        class_name = sequence[:-4]

        files = os.listdir(sequence_path)
        images = [image for image in files if image.endswith('.jpg')]
        images.sort()

        occlusion_file = os.path.join(sequence_path, 'occlusion.txt')
        # patch:
        if sequence == 'helicopter_011':
            occlusions = [False] * 300
        else:
            occlusions = []
            for line in open(occlusion_file):
                line = line.strip()
                if len(line) == 0:
                    continue
                occlusions.append(bool(int(line)))

        groundtruth_file = os.path.join(sequence_path, 'groundtruth.txt')
        bounding_boxes = np.loadtxt(groundtruth_file, dtype=np.int64, delimiter=' ')

        if class_name == 'basketball':
            class_name = 'person'
        elif class_name == 'gymnastics':
            class_name = 'person'
        elif class_name == 'handball':
            class_name = 'person'
        elif class_name == 'hat':
            class_name = 'face'
        elif class_name == 'interview':
            class_name = 'face'
        elif class_name == 'mask':
            class_name = 'face'
        elif class_name == 'motorcycle':
            class_name = 'motorcycle person'
        elif class_name == 'pedestrian':
            class_name = 'person'
        elif class_name == 'politician':
            class_name = 'face'
        elif class_name == 'racing':
            class_name = 'person'
        elif class_name == 'soccer':
            class_name = 'person'
        elif class_name == 'sunglasses':
            class_name = 'face'
        elif class_name == 'tennis':
            class_name = 'person'
        elif sequence == 'long_seq_001':
            class_name = 'person'
        elif sequence == 'long_seq_002':
            class_name = 'car'
        elif sequence == 'long_seq_003':
            class_name = 'airplane'
        elif sequence == 'long_seq_004':
            class_name = 'person'
        elif sequence == 'long_seq_005':
            class_name = 'person'
        assert len(occlusions) == len(bounding_boxes)
        assert len(bounding_boxes) == len(images)

        with constructor.new_sequence(category_name_id_map[class_name]) as sequence_constructor:
            sequence_constructor.set_name(sequence)

            for image, bounding_box, occlusion in zip(images, bounding_boxes, occlusions):
                image_path = os.path.join(sequence_path, image)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                    frame_constructor.set_bounding_box(bounding_box.tolist(), validity=not occlusion)
