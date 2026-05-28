import numpy as np
import os

from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


def construct_ALOV300pp(constructor: SingleObjectTrackingDatasetConstructor, seed):
    annotation_path = seed.annotation_path
    path = seed.root_path

    attributes = os.listdir(path)

    tasks = []
    for attribute in attributes:
        attribute_path = os.path.join(path, attribute)
        attribute_annotation_path = os.path.join(annotation_path, attribute)
        sequences = os.listdir(attribute_path)
        for sequence in sequences:
            sequence_path = os.path.join(attribute_path, sequence)
            sequence_annotation_path = os.path.join(attribute_annotation_path, f'{sequence}.ann')
            tasks.append((sequence, sequence_path, sequence_annotation_path))
    constructor.set_total_number_of_sequences(len(tasks))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for sequence, sequence_path, sequence_annotation_path in tasks:
        images = os.listdir(sequence_path)
        images.sort()
        images = [image for image in images if image.endswith('.jpg')]

        sequence_annotations = np.loadtxt(sequence_annotation_path, dtype=np.float64, delimiter=' ')
        sequence_annotations -= 1

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_path, image))

            for annotation in sequence_annotations:
                frame_index = int(annotation[0].item())
                ax, ay, bx, by, cx, cy, dx, dy = annotation[1:]
                x1 = min(ax, min(bx, min(cx, dx)))
                y1 = min(ay, min(by, min(cy, dy)))
                x2 = max(ax, max(bx, max(cx, dx)))
                y2 = max(ay, max(by, max(cy, dy)))

                with sequence_constructor.open_frame(frame_index) as frame_constructor:
                    frame_constructor.set_bounding_box((x1, y1, x2, y2))
