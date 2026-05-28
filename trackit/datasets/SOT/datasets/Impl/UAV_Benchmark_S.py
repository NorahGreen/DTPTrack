import os
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor


def construct_UAVBenchmarkS(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    annotation_path = seed.annotation_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence)) and sequence != 'UAV-benchmark-SOT_v1.0']
    sequences.sort()
    if annotation_path is None:
        groundtruth_path = os.path.join(root_path, 'UAV-benchmark-SOT_v1.0', 'anno')
    else:
        groundtruth_path = annotation_path

    constructor.set_total_number_of_sequences(len(sequences))

    for sequence in sequences:
        sequence_path = os.path.join(root_path, sequence)
        images = os.listdir(sequence_path)
        images.sort()
        gt_file = '{}_gt.txt'.format(sequence)
        gt_file = os.path.join(groundtruth_path, gt_file)
        bounding_boxes = []
        for line in open(gt_file):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(',')
            assert len(words) == 4
            bounding_box = [int(words[0]) - 1, int(words[1]) - 1, int(words[2]), int(words[3])]
            bounding_boxes.append(bounding_box)

        assert len(images) == len(bounding_boxes)

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)

            for image, bounding_box in zip(images, bounding_boxes):
                image_path = os.path.join(sequence_path, image)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                    frame_constructor.set_bounding_box(bounding_box)
