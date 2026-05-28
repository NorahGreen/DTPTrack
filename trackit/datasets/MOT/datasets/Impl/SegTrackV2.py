import os
from PIL import Image
import numpy as np
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor


def construct_SegTrackV2(construct: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    sequences_frames_path = os.path.join(root_path, 'JPEGImages')
    sequences_annotations_path = os.path.join(root_path, 'GroundTruth')

    sequences = os.listdir(sequences_frames_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(sequences_frames_path, sequence))]
    sequences.sort()

    construct.set_total_number_of_sequences(len(sequences))
    construct.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for sequence in sequences:
        sequence_frames_path = os.path.join(sequences_frames_path, sequence)

        frames = os.listdir(sequence_frames_path)
        frames = [frame for frame in frames if frame.endswith(('.bmp', '.png'))]
        frames.sort()

        sequence_annotation_path = os.path.join(sequences_annotations_path, sequence)
        annotation_files = os.listdir(sequence_annotation_path)
        is_multipart = [os.path.isdir(os.path.join(sequence_annotation_path, annotation_file)) for annotation_file in annotation_files]
        if all(is_multipart):
            annotation_paths = annotation_files
            annotation_file_paths = []
            annotation_paths.sort()
            for annotation_path in annotation_paths:
                annotation_path = os.path.join(sequence_annotation_path, annotation_path)
                annotation_files = os.listdir(annotation_path)
                annotation_files.sort()
                annotation_files = [os.path.join(annotation_path, annotation_file) for annotation_file in annotation_files]
                annotation_file_paths.append(annotation_files)
        else:
            assert not any(is_multipart)

            annotation_files.sort()
            annotation_files = [os.path.join(sequence_annotation_path, annotation_file) for annotation_file in annotation_files]
            if sequence == 'worm':
                annotation_files = annotation_files[1:]
            annotation_file_paths = [annotation_files]

        with construct.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            for frame in frames:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_frames_path, frame))

            for index_of_track, track_annotation_files in enumerate(annotation_file_paths):
                assert len(track_annotation_files) == len(frames)
                sequence_constructor.new_object(index_of_track)
                valid_track_count = 0
                for index_of_frame, track_annotation_file in enumerate(track_annotation_files):
                    ground_truth = Image.open(track_annotation_file)
                    ground_truth = np.asarray(ground_truth)
                    if ground_truth.shape[2] == 3:
                        target_object_pixel_indices = np.where((ground_truth == np.array([255, 255, 255])).all(axis=2))
                    elif ground_truth.shape[2] == 4:
                        target_object_pixel_indices = np.where((ground_truth == np.array([255, 255, 255, 255])).all(axis=2))
                    else:
                        raise RuntimeError(f'Unknown format {ground_truth}')
                    if len(target_object_pixel_indices[0]) != 0 and len(target_object_pixel_indices[1]) != 0:
                        valid_track_count += 1
                        x1 = target_object_pixel_indices[1].min().item()
                        x2 = target_object_pixel_indices[1].max().item()
                        y1 = target_object_pixel_indices[0].min().item()
                        y2 = target_object_pixel_indices[0].max().item()

                        with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                            with frame_constructor.new_object(index_of_track) as object_constructor:
                                object_constructor.set_bounding_box((x1, y1, x2, y2))
                    else:
                        assert not np.any(ground_truth > 0)
                assert valid_track_count != 0
