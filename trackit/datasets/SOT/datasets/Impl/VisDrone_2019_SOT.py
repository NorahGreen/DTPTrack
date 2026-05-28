import os
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
import numpy as np


def construct_VisDrone_2019_SOT(constructor: SingleObjectTrackingDatasetConstructor, seed):
    sequences_path = seed.root_path
    annotations_path = seed.annotations_path

    sequences = os.listdir(sequences_path)
    sequences.sort()
    constructor.set_total_number_of_sequences(len(sequences))
    for sequence in sequences:
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            sequence_path = os.path.join(sequences_path, sequence)
            image_file_names = os.listdir(sequence_path)
            image_file_names.sort()
            for image_file_name in image_file_names:
                image_path = os.path.join(sequence_path, image_file_name)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
            annotation_file_name = sequence + '.txt'
            annotation_file_path = os.path.join(annotations_path, annotation_file_name)
            bounding_boxes = np.loadtxt(annotation_file_path, dtype=np.int64, delimiter=',')
            bounding_boxes[:, 0:2] -= 1
            for index, bounding_box in enumerate(bounding_boxes):
                with sequence_constructor.open_frame(index) as frame_constructor:
                    frame_constructor.set_bounding_box(bounding_box.tolist())
