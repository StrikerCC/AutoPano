# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/9/21 1:40 PM
"""
import os
import pyheif
from PIL import Image


def get_robots_ox_ac_uk_vgg_data(dataset_dir_path):
    """Affine Covariant Features Datasets"""
    img_paths = []
    homo_paths = []
    for file_name in os.listdir(dataset_dir_path):
        if '.bmp' in file_name or 'ppm' in file_name or 'jpg' in file_name or 'pgm' in file_name:
            img_paths.append(dataset_dir_path + '/' + file_name)
        else:
            homo_paths.append(dataset_dir_path + '/' + file_name)
    img_paths.sort(), homo_paths.sort()
    return img_paths, homo_paths


def get_iphone_data(dataset_dir_path):
    img_paths = []
    for file_name in os.listdir(dataset_dir_path):
        if '.bmp' in file_name or \
                'ppm' in file_name or \
                'JPG' in file_name or \
                'jpg' in file_name or \
                'pgm' in file_name:
            img_paths.append(dataset_dir_path + '/' + file_name)
    img_paths.sort()
    return img_paths


def get_iphone_data_heic(dataset_dir_path):
    img_paths = []
    img_names = os.listdir(dataset_dir_path)
    img_names_set = set(img_names)
    for file_name in os.listdir(dataset_dir_path):
        if 'heic' in file_name or \
                'HEIC' in file_name:

            file_name_jpg = file_name[:-4] + 'jpg'
            if file_name_jpg not in img_names_set:
                # reading
                heif_file = pyheif.read(dataset_dir_path + '/' + file_name)

                # convert
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, 'raw', heif_file.mode,
                                      heif_file.stride)

                # saving
                img.save(dataset_dir_path + '/' + file_name_jpg, 'JPEG')

            # record
            img_paths.append(dataset_dir_path + '/' + file_name_jpg)
    img_paths.sort()
    return img_paths


def main():
    # ############################## robots_ox_ac_uk_vgg_data ##############################
    # img_paths, homo_paths = get_robots_ox_ac_uk_vgg_data('../data/graf')
    # print('############################## robots_ox_ac_uk_vgg_data ##############################')
    # print(img_paths)
    # print(homo_paths)
    # print()
    #
    # ############################## iphone ##############################
    # img_paths = get_iphone_data('../data/head')
    # print('############################## iphone ##############################')
    # print(img_paths)
    # print()

    ############################## iphone raw ##############################
    img_paths = get_iphone_data_heic('../data/tank')
    print('############################## iphone ##############################')
    print(img_paths)
    print()


if __name__ == '__main__':
    main()
