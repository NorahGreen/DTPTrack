import os
import csv
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat

_category_id_name_map = {0: 'Car', 1: 'Cyclist', 2: 'Misc', 3: 'Pedestrian', 4: 'Person', 5: 'Tram', 6: 'Truck', 7: 'Van'}


def construct_KITTI_Tracking(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    images_path = seed.root_path
    labels_path = seed.label_path
    labels_path = os.path.join(labels_path, 'training', 'label_02')

    sequences = os.listdir(images_path)
    sequences.sort()
    constructor.set_total_number_of_sequences(len(sequences))
    constructor.set_category_id_name_map(_category_id_name_map)
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}
    for sequence in sequences:
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            # KITTI pipelines benchmark data format:
            # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            sequence_images_path = os.path.join(images_path, sequence)
            images = os.listdir(sequence_images_path)
            images = [image for image in images if image.endswith('.png')]
            images.sort()

            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_images_path, image))

            tracklet_ids = {}

            annotation_file_path = os.path.join(labels_path, sequence + '.txt')
            with open(annotation_file_path, 'r') as fid:
                reader = csv.reader(fid, delimiter=' ')
                for row in reader:
                    object_type = row[2]
                    if object_type == 'DontCare':
                        continue
                    tracklet_id = int(row[1])
                    if tracklet_id not in tracklet_ids:
                        tracklet_ids[tracklet_id] = object_type
                        with sequence_constructor.new_object(tracklet_id) as object_constructor:
                            object_constructor.set_category_id(category_name_id_map[object_type])
                    else:
                        assert tracklet_ids[tracklet_id] == object_type
                    frame_index = int(row[0])
                    truncation = int(row[3])
                    occlusion = int(row[4])
                    observation_angle = float(row[5])
                    x1 = float(row[6])
                    y1 = float(row[7])
                    x2 = float(row[8])
                    y2 = float(row[9])
                    height_m = float(row[10])
                    width_m = float(row[11])
                    length_m = float(row[12])
                    x_m = float(row[13])
                    y_m = float(row[14])
                    z_m = float(row[15])
                    yaw_angle = float(row[16])
                    bounding_box = [x1, y1, x2, y2]
                    is_present = not(truncation == 2 or occlusion == 2)
                    with sequence_constructor.open_frame(frame_index) as frame_constructor:
                        with frame_constructor.new_object(tracklet_id) as object_constructor:
                            object_constructor.set_bounding_box(bounding_box, validity=is_present)
