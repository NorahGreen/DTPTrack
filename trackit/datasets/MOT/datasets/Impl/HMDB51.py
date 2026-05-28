import os
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.datasets.base.tool.parse_THOTH_bb_file import parse_THOTH_bb_file
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


# https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
# http://lear.inrialpes.fr/people/wang/improved_trajectories
def construct_HMDB51(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    annotation_path = seed.annotation_path
    actions = os.listdir(root_path)
    actions.sort()

    sequences = []
    constructor.set_category_id_name_map({0: 'person'})
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for action in actions:
        action_path = os.path.join(root_path, action)
        videos = os.listdir(action_path)
        videos.sort()
        for video in videos:
            video_path = os.path.join(action_path, video)
            images = os.listdir(video_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            annotation_file_path = os.path.join(annotation_path, video + '.bb')
            frame_annotations = parse_THOTH_bb_file(annotation_file_path)
            if len(frame_annotations) == 0:
                continue
            sequences.append((action, video, video_path, images, frame_annotations))

    constructor.set_total_number_of_sequences(len(sequences))
    for action_name, video_name, video_path, images, frame_annotations in sequences:
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(video_name)
            sequence_constructor.set_attribute('action', action_name)
            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(video_path, image))
            registered_object_ids = {}
            for frame_index, object_annotations in frame_annotations.items():
                for object_id, (bounding_box, confidence_score) in object_annotations.items():
                    if object_id not in registered_object_ids:
                        with sequence_constructor.new_object(object_id) as object_constructor:
                            object_constructor.set_category_id(0)

                        registered_object_ids[object_id] = None
                    with sequence_constructor.open_frame(frame_index) as frame_constructor:
                        with frame_constructor.new_object(object_id) as object_constructor:
                            object_constructor.set_bounding_box(bounding_box)
                            object_constructor.set_attribute('confidence', confidence_score)
