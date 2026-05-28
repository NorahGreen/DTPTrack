import os
import numpy as np
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


def get_class_name(sequence: str):
    class_name = []
    for c in sequence:
        if c.isdigit():
            continue
        c = c.lower()
        class_name.append(c)
    return ''.join(class_name)


_category_id_name_map = {0: 'airbattle', 1: 'aquaplane', 2: 'avatar', 3: 'backkom', 4: 'bike', 5: 'bluecar', 6: 'boarding', 7: 'bolt', 8: 'carcorder', 9: 'carscale', 10: 'circle', 11: 'cliff-dive', 12: 'dancer', 13: 'diving', 14: 'drift', 15: 'eagle', 16: 'fighters', 17: 'flying', 18: 'football', 19: 'game', 20: 'gymnastics', 21: 'helicopter', 22: 'horse', 23: 'horsemanship', 24: 'jump', 25: 'kwan', 26: 'larva', 27: 'lemming', 28: 'lipinski', 29: 'lola', 30: 'mario', 31: 'monkey', 32: 'neymar', 33: 'pole-dance', 34: 'rcplane', 35: 'roboone', 36: 'robotgoogle', 37: 'run', 38: 'suneo', 39: 'torus', 40: 'trampoline', 41: 'transformer', 42: 'trellis', 43: 'uneven-bars', 44: 'up', 45: 'waterski', 46: 'yunakim'}


def construct_DeformSOT(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence)) and sequence != 'Annotations']

    groundtruth_folder = os.path.join(root_path, 'Annotations', 'Base')
    constructor.set_total_number_of_sequences(len(sequences))
    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}
    for sequence in sequences:
        sequence_dir = os.path.join(root_path, sequence)
        gt_file = os.path.join(groundtruth_folder, '{}.txt'.format(sequence))

        bounding_boxes = np.loadtxt(gt_file, dtype=np.int64, delimiter=',')
        class_label = get_class_name(sequence)
        images = os.listdir(sequence_dir)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        with constructor.new_sequence(category_name_id_map[class_label]) as sequence_constructor:
            sequence_constructor.set_name(sequence)
            for image, bounding_box in zip(images, bounding_boxes):
                image_path = os.path.join(sequence_dir, image)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                    frame_constructor.set_bounding_box(bounding_box.tolist())
