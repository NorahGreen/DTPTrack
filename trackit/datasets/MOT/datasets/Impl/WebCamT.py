import os
from trackit.datasets.MOT.constructor import MultipleObjectTrackingDatasetConstructor
from trackit.miscellanies.most_frequent import get_most_frequent_item_from_list


def construct_WebCamT(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    include_passenger = seed.include_passenger
    data_split = seed.data_split

    class_names = {}
    for idx, name in enumerate(open(os.path.join(root_path, 'city_cam.names'))):
        name = name.strip()
        class_names[idx] = name
    if include_passenger:
        class_names[len(class_names)] = 'passenger'

    constructor.set_category_id_name_map(class_names)

    def _parseTrainTestSeparationFile(path):
        paths = []
        for line in open(path, 'r'):
            line = line.strip()
            if len(line) > 0:
                words = line.split('-')
                paths.append((words[0], line))
        return paths

    def _getWidthAndHeight(file_content: str, begin_index: int):
        width_begin_index = file_content.find('<width>', begin_index)
        if width_begin_index == -1:
            return None
        width_end_index = file_content.find('</width>', width_begin_index)
        height_begin_index = file_content.find('<height>', width_end_index)

        height_end_index = file_content.find('</height>', height_begin_index)

        width = int(file_content[width_begin_index + 7: width_end_index])
        height = int(file_content[height_begin_index + 8: height_end_index])

        return width, height, height_end_index + 9

    def _findNextObject(file_content: str, begin_index: int):
        vehicle_begin = file_content.find('<vehicle>', begin_index)
        passenger_begin = file_content.find('<passengers>', begin_index)

        if vehicle_begin == -1 and passenger_begin == -1:
            return None

        if vehicle_begin == -1:
            is_vehicle = False
        elif passenger_begin == -1:
            is_vehicle = True
        elif vehicle_begin < passenger_begin:
            is_vehicle = True
        else:
            is_vehicle = False

        if is_vehicle:
            begin_index = vehicle_begin + 9
        else:
            begin_index = passenger_begin + 12

        id_begin_index = file_content.find('<id>', begin_index)
        id_begin_index += len('<id>')
        id_end_index = file_content.find('</id>', id_begin_index)

        id_ = file_content[id_begin_index: id_end_index]
        id_end_index += len('</id>')

        xmax_begin_index = file_content.find('<xmax>', id_end_index)
        xmax_end_index = file_content.find('</xmax>', xmax_begin_index)
        xmin_begin_index = file_content.find('<xmin>', xmax_end_index)

        xmin_end_index = file_content.find('</xmin>', xmin_begin_index)
        ymax_begin_index = file_content.find('<ymax>', xmin_end_index)
        ymax_end_index = file_content.find('</ymax>', ymax_begin_index)
        ymin_begin_index = file_content.find('<ymin>', ymax_end_index)
        ymin_end_index = file_content.find('</ymin>', ymin_begin_index)
        if is_vehicle:
            type_begin_index = file_content.find('<type>', ymin_end_index)
            type_end_index = file_content.find('</type>', type_begin_index)
            end_index = type_end_index + 7
        else:
            end_index = ymin_end_index + 7

        xmax = int(file_content[xmax_begin_index + 6: xmax_end_index])
        xmin = int(file_content[xmin_begin_index + 6: xmin_end_index])
        ymax = int(file_content[ymax_begin_index + 6: ymax_end_index])
        ymin = int(file_content[ymin_begin_index + 6: ymin_end_index])
        if is_vehicle:
            object_category = int(file_content[type_begin_index + 6: type_end_index]) - 1
        else:
            object_category = len(class_names) - 1

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        return id_, is_vehicle, object_category, x, y, w, h, end_index

    train_paths = []
    val_paths = []

    train_paths += _parseTrainTestSeparationFile(
        os.path.join(root_path, 'train_test_separation', 'Downtown_Train.txt'))
    train_paths += _parseTrainTestSeparationFile(
        os.path.join(root_path, 'train_test_separation', 'Parkway_Train.txt'))

    val_paths += _parseTrainTestSeparationFile(
        os.path.join(root_path, 'train_test_separation', 'Downtown_Test.txt'))
    val_paths += _parseTrainTestSeparationFile(
        os.path.join(root_path, 'train_test_separation', 'Parkway_Test.txt'))

    paths = []
    if 'train' in data_split:
        paths += train_paths
    if 'val' in data_split:
        paths += val_paths

    constructor.set_total_number_of_sequences(len(paths))

    for path in paths:
        with constructor.new_sequence() as sequence_constructor:
            sequence_name = path[1]
            sequence_constructor.set_name(sequence_name)
            xml_files = os.listdir(os.path.join(root_path, *path))
            xml_files = [file_name for file_name in xml_files if file_name.endswith('.xml')]
            xml_files.sort()

            vehicle_id_real_id_mapper = {}
            passenger_id_real_id_mapper = {}
            real_id = 0
            id_object_category_mapper = {}

            index_of_frame = -1

            for xml_file in xml_files:
                with open(os.path.join(root_path, *path, xml_file), 'r') as fid:
                    file_content = fid.read()

                image_file_name = xml_file[:-3] + 'jpg'
                image_path = os.path.join(root_path, *path, image_file_name)
                if not os.path.exists(image_path):
                    print('Image file: {} not exists'.format(image_path))
                    continue

                index_of_frame += 1
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)

                begin_index = 0
                image_size = _getWidthAndHeight(file_content, begin_index)
                if image_size is None:
                    print('Invalid annotation file: {}'.format(os.path.join(*path, xml_file)))
                    continue

                width, height, begin_index = image_size

                while True:
                    vehicle = _findNextObject(file_content, begin_index)
                    if vehicle is None:
                        break
                    id_, is_vehicle, category, x, y, w, h, begin_index = vehicle

                    if not include_passenger:
                        if not is_vehicle:
                            continue

                    def _printInvalidBoundingBoxAndFixed():
                        print('Invalid bounding box in annotation file: {}. Fixed.'.format(os.path.join(*path, xml_file)))
                        print((x, y, w, h))

                    def _printInvalidBoundingBox():
                        print('Invalid bounding box in annotation file: {}.'.format(os.path.join(*path, xml_file)))
                        print((x, y, w, h))

                    if w <= 0 or h <= 0:
                        _printInvalidBoundingBox()
                        continue

                    if x < 0:
                        _printInvalidBoundingBoxAndFixed()
                        x = 0

                    if y < 0:
                        _printInvalidBoundingBoxAndFixed()
                        y = 0

                    if x + w >= width:
                        _printInvalidBoundingBoxAndFixed()
                        w = width - x - 1

                    if y + h >= height:
                        _printInvalidBoundingBoxAndFixed()
                        h = height - y - 1

                    if is_vehicle:
                        if category < 0 or category >= len(class_names) - 1:
                            print('Invalid type in annotation file: {}'.format(os.path.join(*path, xml_file)))
                            continue

                    if is_vehicle:
                        if id_ not in vehicle_id_real_id_mapper:
                            vehicle_id_real_id_mapper[id_] = real_id
                            current_object_id = real_id
                            real_id += 1

                            id_object_category_mapper[current_object_id] = [category]
                        else:
                            current_object_id = vehicle_id_real_id_mapper[id_]
                            id_object_category_mapper[current_object_id].append(category)
                    else:
                        if id_ not in passenger_id_real_id_mapper:
                            passenger_id_real_id_mapper[id_] = real_id
                            current_object_id = real_id
                            real_id += 1

                            with sequence_constructor.new_object(current_object_id) as object_constructor:
                                object_constructor.set_category_id(category)
                        else:
                            current_object_id = passenger_id_real_id_mapper[id_]

                    with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                        with frame_constructor.new_object(current_object_id) as object_constructor:
                            object_constructor.set_bounding_box((x, y, w, h))

            for object_id, category_ids in id_object_category_mapper.items():
                category_id = get_most_frequent_item_from_list(category_ids)

                with sequence_constructor.new_object(object_id) as object_constructor:
                    object_constructor.set_category_id(category_id)
