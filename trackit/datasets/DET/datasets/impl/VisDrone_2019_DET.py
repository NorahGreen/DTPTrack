from trackit.datasets.DET.constructor import DetectionDatasetConstructor
import os


def construct_VisDrone2019_DET(constructor: DetectionDatasetConstructor, seed):
    images_path = seed.root_path
    annotation_path = seed.annotation_path
    images = os.listdir(images_path)
    images.sort()

    categories = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
    constructor.set_category_id_name_map({index: name for index, name in enumerate(categories)})
    constructor.set_total_number_of_images(len(images))
    for image_file_name in images:
        image_name = image_file_name[: -4]
        with constructor.new_image() as image_constructor:
            image_constructor.set_path(os.path.join(images_path, image_file_name))

            annotation_file_name = image_name + '.txt'
            '''
            <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    
                Name                                                  Description
            -------------------------------------------------------------------------------------------------------------------------------     
             <bbox_left>	     The x coordinate of the top-left corner of the predicted bounding box
    
             <bbox_top>	     The y coordinate of the top-left corner of the predicted object bounding box
    
             <bbox_width>	     The width in pixels of the predicted object bounding box
    
            <bbox_height>	     The height in pixels of the predicted object bounding box
    
               <score>	     The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                                 an object instance.
                                 The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
                                 while 0 indicates the bounding box will be ignored.
    
            <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
                                 people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
                                 others(11))
    
            <truncation>	     The score in the DETECTION result file should be set to the constant -1.
                                 The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                                 (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
    
            <occlusion>	     The score in the DETECTION file should be set to the constant -1.
                                 The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
                                 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
                                 (occlusion ratio 50% ~ 100%)).
            '''
            for line in open(os.path.join(annotation_path, annotation_file_name), 'r', encoding='utf-8'):
                line = line.strip()
                if len(line) == 0:
                    continue
                words = line.split(',')
                words = [word for word in words if len(word) > 0]
                assert len(words) == 8
                object_category = int(words[5])
                truncation = int(words[6])
                occlusion = int(words[7])
                with image_constructor.new_object() as object_constructor:
                    object_constructor.set_bounding_box([int(words[0]) - 1, int(words[1]) - 1, int(words[2]), int(words[3])], validity=occlusion != 2)
                    object_constructor.set_category_id(object_category)
                    object_constructor.merge_attributes({'truncation': truncation, 'occlusion': occlusion})
