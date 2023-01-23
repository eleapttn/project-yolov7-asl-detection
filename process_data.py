import os
import shutil
import albumentations as A
import cv2
import yaml
import matplotlib.pyplot as plt

# allow a directory creation even if it already exists
def create_directories(new_directory):
    
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
    os.makedirs(new_directory)
    
    return new_directory

# extract classes names from the data.yaml file
def yaml_names():
    
    # open data.yaml file
    with open(r'/workspace/data/data_base/data.yaml') as file:
        
        # load the full yaml file and extract the list of names
        parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)
        list_class_names = parsed_yaml_file.get('names')
      
    return list_class_names


# convert text files to lists and define categories of bboxes
def bboxes_classes_extraction(txt_to_list, class_names):
    
    bboxes = []
    category_ids = []
    category_id_to_name = {}
    
    for bb in txt_to_list:
        bb = [eval(i) for i in bb]
        category_ids.append(bb[0])
        category_id_to_name[bb[0]] = class_names[bb[0]]
        bb.pop(0)
        bboxes.append(bb)
        print("bboxes:", bboxes)
        print("category_ids:", category_ids)
        print("category_id_to_name:", category_id_to_name)
    
    return bboxes, category_ids, category_id_to_name


# convert lists into text files
def txt_labels_creation(new_bboxes, new_category_ids, dest_path, f_name):

    bbxs = []
    for i in range(len(new_bboxes)):

        list_class_bb = list(new_bboxes[i])
        list_class_bb.insert(0, new_category_ids[i])
        list_class_bb = [str(x) for x in list_class_bb]
        bbxs.append(list_class_bb)

    with open(f"/workspace/data/train_data_augmentation/labels/{dest_path}/{f_name}_{dest_path}.txt", "w") as list_to_txt:
        for l in bbxs:
            list_to_txt.write(" ".join(l))

    list_to_txt.close()
    
    return list_to_txt


# process horizontal flip
def data_transform(source_path, dest_path, list_class_names):

    # create the new destination path for images
    dest_path_img = f"/workspace/data/train_data_augmentation/images/{dest_path}"
    create_directories(dest_path_img)
    
    dest_path_lbl = f"/workspace/data/train_data_augmentation/labels/{dest_path}"
    create_directories(dest_path_lbl)
    
    print(dest_path)
    
    # put randomly N images and corresponding labels in the destination paths
    for img in os.listdir(source_path):

        # image to process
        image = cv2.imread(f"{source_path}/{img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # text to process
        f_name = img.removesuffix('.jpg')
        print(f_name)
        labels = os.path.join(f"/workspace/data/train_data_to_process/labels/{dest_path}/", f"{f_name}.txt")
        txt_file = open(labels, 'r')

        # text to list
        txt_to_list = [(line.strip()).split() for line in txt_file]

        # extraction of needed information
        bboxes, category_ids, category_id_to_name = bboxes_classes_extraction(txt_to_list, list_class_names)

        # data transformation with the corresponding method
        if dest_path=="horizontal_flip":
            transformation = transform_horizontal_flip(image=image, bboxes=bboxes, category_ids=category_ids)

        elif dest_path=="rotation":
            transformation = transform_rotation(image=image, bboxes=bboxes, category_ids=category_ids)

        elif dest_path=="saturation":
            transformation = transform_saturation(image=image, bboxes=bboxes, category_ids=category_ids)

        elif dest_path=="brightness":
            transformation = transform_brightness(image=image, bboxes=bboxes, category_ids=category_ids)

        elif dest_path=="translation_gray":
            transformation = transform_translation_gray(image=image, bboxes=bboxes, category_ids=category_ids)

        transformed_image = transformation['image']
        transformed_bboxes = transformation['bboxes']
        transformed_category_ids = transformation['category_ids']

        plt.imshow(transformed_image)
        plt.savefig(f"/workspace/data/train_data_augmentation/images/{dest_path}/{f_name}_{dest_path}.jpg")

        txt_labels_creation(transformed_bboxes, transformed_category_ids, dest_path, f_name)

        print(f_name)
        
    return


# main
if __name__ == '__main__':
    
    # define transformations
    transform_horizontal_flip = A.Compose([
        A.HorizontalFlip(p=0.8),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    
    transform_rotation = A.Compose([
        A.augmentations.geometric.rotate.Rotate(60),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    transform_saturation = A.Compose([
        A.CLAHE(p=0.8),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    
    transform_brightness = A.Compose([
        A.RandomBrightnessContrast(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    
    transform_translation_gray = A.Compose([
        A.Affine(0.2),
        A.ToGray(always_apply=True, p=0.8),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    
    # define method
    method = "translation_gray"

    # define the path to access to the training set to be processed
    source_path = f"/workspace/data/train_data_to_process/images/{method}"

    # extract class names
    list_class_names = yaml_names()

    # apply `data_transform` function for the five cases
    data_transform(source_path, method, list_class_names)