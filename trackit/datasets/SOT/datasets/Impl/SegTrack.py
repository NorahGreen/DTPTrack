from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat
import os
from PIL import Image
import numpy as np


def construct_SegTrack(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path

    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)
    sequences = os.listdir(root_path)
    sequences.sort()
    constructor.set_total_number_of_sequences(len(sequences))
    for sequence in sequences:
        sequence_path = os.path.join(root_path, sequence)
        images = os.listdir(sequence_path)
        images = [image for image in images if image.endswith('.png') or image.endswith('.bmp')]
        images.sort()

        ground_truth_images_path = os.path.join(sequence_path, 'ground-truth')
        if not os.path.exists(ground_truth_images_path):
            ground_truth_images_path = os.path.join(sequence_path, 'ground_truth')
        ground_truth_images = os.listdir(ground_truth_images_path)
        ground_truth_images.sort()
        assert len(images) == len(ground_truth_images)

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            for image, ground_truth_image in zip(images, ground_truth_images):
                image_path = os.path.join(sequence_path, image)
                ground_truth_image_path = os.path.join(ground_truth_images_path, ground_truth_image)
                ground_truth = Image.open(ground_truth_image_path)
                ground_truth = np.asarray(ground_truth)
                target_object_pixel_indices = np.where((ground_truth == np.array([255, 255, 255])).all(axis=2))
                x1 = target_object_pixel_indices[1].min().item()
                x2 = target_object_pixel_indices[1].max().item()
                y1 = target_object_pixel_indices[0].min().item()
                y2 = target_object_pixel_indices[0].max().item()
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                    frame_constructor.set_bounding_box((x1, y1, x2, y2))
