import os
import random
import shutil

# allow a directory creation even if it already exists
def create_directories(new_directory):
    
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
    os.makedirs(new_directory)
    
    return new_directory

# split train data in five in order to process it
def split_train_data(files, source_path, dest_path):
    
    # split in 5
    N = len(files) // 5
    
    # create the new destination path for images
    dest_path_img = f"/workspace/data/train_data_to_process/images/{dest_path}"
    create_directories(dest_path_img)

    # create the new destination path for labels
    dest_path_lbl = f"/workspace/data/train_data_to_process/labels/{dest_path}"
    create_directories(dest_path_lbl)
    
    # put randomly N images and corresponding labels in the destination paths
    for f in random.sample(files, N):
        # copy image
        shutil.copy(os.path.join(source_path, f), dest_path_img)
        
        # copy the corresponding label (txt file)
        f_name = f.removesuffix('.jpg')
        shutil.copy(os.path.join('/workspace/data/data_base/train/labels', f"{f_name}.txt"), dest_path_lbl)
    
    # obtain N images and labels in each category
    print(N, "the number of success transfers")
    
    return

# main
if __name__ == '__main__':
    
    # define the path to access to the basic training set
    source_path = '/workspace/data/data_base/train/images'

    # apply `split_train_data` function for the five cases
    split_train_data(os.listdir(source_path), source_path, 'horizontal_flip')
    split_train_data(os.listdir(source_path), source_path, 'rotation')
    split_train_data(os.listdir(source_path), source_path, 'saturation')
    split_train_data(os.listdir(source_path), source_path, 'brightness')
    split_train_data(os.listdir(source_path), source_path, 'translation_gray')