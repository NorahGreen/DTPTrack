import os
from typing import Dict, List, Tuple
import csv
from PIL import Image
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.datasets.common.types.bounding_box import BoundingBoxFormat


class YoutubeBBAnnotationEntry:
    youtube_id: str
    time_stamp: int
    class_id: int
    class_name: str
    object_id: int
    is_present: str
    xmin_ratio: float
    xmax_ratio: float
    ymin_ratio: float
    ymax_ratio: float


YoutubeBBClassIndexNameMapper = {0: 'person', 1: 'bird', 2: 'bicycle', 3: 'boat', 4: 'bus', 5: 'bear', 6: 'cow', 7: 'cat',
                        8: 'giraffe', 9: 'potted plant', 10: 'horse', 11: 'motorcycle', 12: 'knife', 13: 'airplane',
                        14: 'skateboard', 15: 'train', 16: 'truck', 17: 'zebra', 18: 'toilet', 19: 'dog',
                        20: 'elephant', 21: 'umbrella', 23: 'car'}


def parseYoutubeBBCSV(file_path: str):
    annotations = []
    with open(file_path, 'r') as fid:
        reader = csv.reader(fid)
        for row in reader:
            entry = YoutubeBBAnnotationEntry()
            entry.youtube_id = row[0]
            entry.time_stamp = int(row[1])
            entry.class_id = int(row[2])
            entry.class_name = row[3]
            entry.object_id = int(row[4])
            assert YoutubeBBClassIndexNameMapper[entry.class_id] == entry.class_name, 'error: class_id {} and class_name {} mismatch in entry {}_{}'.format(entry.class_id, entry.class_name, entry.youtube_id, entry.object_id)
            entry.is_present = row[5]

            entry.xmin_ratio = float(row[6])
            entry.xmax_ratio = float(row[7])
            entry.ymin_ratio = float(row[8])
            entry.ymax_ratio = float(row[9])

            annotations.append(entry)

    return annotations


def organize_annotations(annotations: List[YoutubeBBAnnotationEntry]):
    #                         ytb_id   time_stamp      obj_id cls_id                      bbox             is_present
    organized_annotations: Dict[str, Dict[int, Dict[Tuple[int, int], Tuple[Tuple[float, float, float, float], str]]]] = {}

    for annotation in annotations:
        if annotation.youtube_id not in organized_annotations:
            organized_annotations[annotation.youtube_id] = {}
        current_ytb_vid = organized_annotations[annotation.youtube_id]
        if annotation.time_stamp not in current_ytb_vid:
            current_ytb_vid[annotation.time_stamp] = {}
        current_frame = current_ytb_vid[annotation.time_stamp]
        object_identity = (annotation.object_id, annotation.class_id)
        assert object_identity not in current_frame
        current_frame[object_identity] = ((annotation.xmin_ratio, annotation.xmax_ratio, annotation.ymin_ratio, annotation.ymax_ratio), annotation.is_present)

    return organized_annotations


def construct_YoutubeBB(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    train_csv = seed.train_csv
    validation_csv = seed.validation_csv

    if not train_csv:
        train_csv = os.path.join(root_path, 'youtube_boundingboxes_detection_train.csv')
    if not validation_csv:
        validation_csv = os.path.join(root_path, 'youtube_boundingboxes_detection_validation.csv')

    data: List[YoutubeBBAnnotationEntry] = []
    if 'train' in data_split:
        data.extend(parseYoutubeBBCSV(train_csv))
    if 'val' in data_split:
        data.extend(parseYoutubeBBCSV(validation_csv))

    annotations = organize_annotations(data)

    constructor.set_total_number_of_sequences(len(annotations))
    constructor.set_category_id_name_map(YoutubeBBClassIndexNameMapper)
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for youtube_id, video_annotations in annotations.items():
        video_path = os.path.join(root_path, youtube_id)
        if not os.path.exists(video_path):
            continue
        registered_object = {}

        image_size = None
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(youtube_id)
            sequence_constructor.set_fps(1)

            index_of_frame = -1
            for time_stamp, object_annotation in video_annotations.items():
                image_path = os.path.join(video_path, '{}.jpg'.format(time_stamp))
                if not os.path.exists(image_path):
                    continue
                if image_size is None:
                    image_size = Image.open(image_path).size

                index_of_frame += 1
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path, image_size)

                for (object_id, class_id), (bounding_box, is_present) in object_annotation.items():
                    if (object_id, class_id) not in registered_object:
                        id_ = len(registered_object)
                        with sequence_constructor.new_object(id_) as object_constructor:
                            object_constructor.set_category_id(class_id)
                        registered_object[(object_id, class_id)] = id_
                    else:
                        id_ = registered_object[(object_id, class_id)]
                    bounding_box = [bounding_box[0] * image_size[0], bounding_box[1] * image_size[0], bounding_box[2] * image_size[1], bounding_box[3] * image_size[1]]
                    bounding_box = [bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3]]
                    with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                        with frame_constructor.new_object(id_) as object_constructor:
                            object_constructor.set_bounding_box(bounding_box, validity=is_present == 'present')
