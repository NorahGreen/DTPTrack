from ..AIC19_Track1 import AIC19_Track1_Seed
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.datasets.base.tool.decode_video_to_path import decode_video_file
import os
import shutil


def construct_AIC19Track1(constructor: MultipleObjectTrackingDatasetConstructor, seed: AIC19_Track1_Seed):
    scenes_path = os.path.join(seed.root_path, 'train')
    scenes = os.listdir(scenes_path)
    scenes.sort()

    sequence_names = []
    sequence_paths = []

    for scene in scenes:
        scene_path = os.path.join(scenes_path, scene)
        cameras = os.listdir(scene_path)
        cameras.sort()
        for camera in cameras:
            camera_path = os.path.join(scene_path, camera)
            images_path = os.path.join(camera_path, 'imgs')
            if not os.path.exists(images_path):
                tmp_path = os.path.join(camera_path, 'tmp')
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)
                os.mkdir(tmp_path)
                decode_video_file(os.path.join(camera_path, 'vdo.avi'), tmp_path)
                os.rename(tmp_path, images_path)
            sequence_paths.append(camera_path)
            sequence_names.append('-'.join((scene, camera)))

    constructor.set_total_number_of_sequences(len(sequence_paths))

    for sequence_name, camera_path in zip(sequence_names, sequence_paths):
        images_path = os.path.join(camera_path, 'imgs')
        images = os.listdir(images_path)
        images.sort()
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)

            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(images_path, image))

            gt_file_path = os.path.join(camera_path, 'Base', 'Base.txt')
            ids = {}
            # [frame, ID, left, top, width, height, 1, -1, -1, -1]
            for line in open(gt_file_path):
                line = line.strip()
                if len(line) == 0:
                    continue
                values = line.split(',')
                assert len(values) == 10
                frame_index = int(values[0]) - 1
                object_id = int(values[1])
                if object_id not in ids:
                    ids[object_id] = None
                    sequence_constructor.new_object(object_id)
                with sequence_constructor.open_frame(frame_index) as frame_constructor:
                    with frame_constructor.new_object(object_id) as object_constructor:
                        object_constructor.set_bounding_box((int(values[2]), int(values[3]), int(values[4]), int(values[5])))
