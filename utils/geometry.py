# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/9/21 5:30 PM
"""
import numpy as np


def xyxy_2_corners_coord(x_min, y_min, x_max, y_max):
    # top_left, top_right, low_right, low_left,
    img_box_coord_tgt = np.array([
        [x_min, y_min, 1],
        [x_max, y_min, 1],
        [x_max, y_max, 1],
        [x_min, y_max, 1],
    ])
    return img_box_coord_tgt


def corners_2_bounding_box_xyxy(img_box_coord_tgt):
    # assert img_box_coord_tgt[0, 0] == img_box_coord_tgt[-1, 0], 'x_min ' + str(img_box_coord_tgt[0, 0]) + ' not equals to ' + str(img_box_coord_tgt[-1, 0])
    # assert img_box_coord_tgt[0, 1] == img_box_coord_tgt[1, 1], 'y_min ' + str(img_box_coord_tgt[0, 1]) + ' not equals to ' + str(img_box_coord_tgt[1, 1])
    # assert img_box_coord_tgt[1, 0] == img_box_coord_tgt[2, 0], 'x_max ' + str(img_box_coord_tgt[1, 0]) + ' not equals to ' + str(img_box_coord_tgt[2, 0])
    # assert img_box_coord_tgt[2, 1] == img_box_coord_tgt[3, 1], 'y_max ' +  str(img_box_coord_tgt[2, 1]) + ' not equals to ' + str(img_box_coord_tgt[3, 1])

    x_min, x_max = np.min(img_box_coord_tgt[:, 0]), np.max(img_box_coord_tgt[:, 0])
    y_min, y_max = np.min(img_box_coord_tgt[:, 1]), np.max(img_box_coord_tgt[:, 1])
    return x_min, y_min, x_max, y_max


def merge_box(boxes):
    box_biggest = boxes[0]
    for box in boxes:
        box_biggest = (min(box_biggest[0], box[0]), min(box_biggest[1], box[1]), max(box_biggest[2], box[2]), max(box_biggest[3], box[3]))
    return box_biggest


def main():
    xyxy = (0, 0, 100, 100)
    corners = xyxy_2_corners_coord(*xyxy)
    print(corners)
    xyxy = corners_2_bounding_box_xyxy(corners)
    print(xyxy)


if __name__ == '__main__':
    main()
