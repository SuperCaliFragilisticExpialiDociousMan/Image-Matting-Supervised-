import cv2
import numpy as np
import os
import matplotlib as mpl
import pylab
mpl.use('TkAgg')

import utils

def get_image_paths(root_dir):
    image_paths_dict = {}
    matting_paths_dict = {}
    for root, dirs, files in os.walk(root_dir):
        if not files:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_path = file_path.replace('\\', '/')
            file_name = file.split('.')[0]
            dir_name = file_path.split('/')[-2]
            if dir_name.startswith('clip'):
                image_paths_dict[file_name] = file_path
            if dir_name.startswith('matting'):
                matting_paths_dict[file_name] = file_path

    image_corresponding_paths = []
    for image_name, path in image_paths_dict.items():
        matting_path = matting_paths_dict.get(image_name, None)
        if matting_path is not None:
            image_corresponding_paths.append([path, matting_path])
        else:
            print(path)
    print('Number of valid images: ', len(image_corresponding_paths))
    if len(image_corresponding_paths) < 1:
        raise ValueError('`root_dir` is error. Please reset it correctly.')
    return image_corresponding_paths


if __name__ == '__main__':
    root_dir = "/Users/langjiedong/Desktop/NEU_Courses/DS5220/Project/code/Code/Demo/Matting_Human_Half";
    train_txt_path = './train.txt'
    val_txt_path = './val.txt'
    image_paths = get_image_paths(root_dir=root_dir)
    print(image_paths[1])
    image = cv2.imread(image_paths[1][0])
    cv2.imshow('Original', image)
    print(image.shape)
    image_ = cv2.resize(image, (300, 400))
    cv2.imshow('Stretched Image', image_)
    cv2.imwrite('test.png', image_)
    pylab.show()
    print(image_.shape)
    #cv2.imshow(image.squeeze(), cmap = 'gray')
    #.squeeze(), cmap = 'gray'