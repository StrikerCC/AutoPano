# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/9/21 2:25 PM
"""

import utils.dataset
# import utils.match_by_feature
import utils.view_geometry
import utils.geometry
import utils.img_blending

import numpy as np
import cv2


def warp_img(img_src, img_tgt, homo_src_2_tgt):
    if img_src is None or img_tgt is None:
        print('Empty image')
        return None

    '''compute expect img size after warp'''
    m, n = img_tgt.shape[:2]
    img_box_tgt = (0, 0, n, m)
    img_corners_tgt = utils.geometry.xyxy_2_corners_coord(*img_box_tgt)

    m, n = img_src.shape[:2]
    img_box_src = (0, 0, n, m)
    img_corners_src = utils.geometry.xyxy_2_corners_coord(*img_box_src)

    del m, n

    # top_left, top_right, low_right, low_left, in xyxy'''
    img_corners_src_warped = np.matmul(homo_src_2_tgt, img_corners_src.T).T
    img_corners_src_warped = img_corners_src_warped / img_corners_src_warped[:, -1:]
    img_box_src_warp = utils.geometry.corners_2_bounding_box_xyxy(img_corners_src_warped)

    img_box_tgt_plus_warped_src = utils.geometry.merge_box([img_box_src_warp, img_box_tgt])

    # determine transformation from warped output to positive output according to img_box_out
    euclidean_tgt_2_out = np.eye(3)
    euclidean_tgt_2_out[0, -1] = -img_box_tgt_plus_warped_src[0]
    euclidean_tgt_2_out[1, -1] = -img_box_tgt_plus_warped_src[1]

    # determine out size and transformation from tgt img to output img according to img_box_out
    out_size = (int(img_box_tgt_plus_warped_src[2]-img_box_tgt_plus_warped_src[0]), int(img_box_tgt_plus_warped_src[3]-img_box_tgt_plus_warped_src[1]))
    homo_tgt_2_out = euclidean_tgt_2_out
    img_corners_tgt_in_out = np.matmul(euclidean_tgt_2_out, img_corners_tgt.T).T
    img_box_tgt_in_out = utils.geometry.corners_2_bounding_box_xyxy(img_corners_tgt_in_out)

    # update the transformation from src im to output img
    homo_src_2_out = np.matmul(homo_tgt_2_out, homo_src_2_tgt)

    # print(img_corners_src)
    # print(img_corners_src_warped)
    img_warped = cv2.warpPerspective(img_src, homo_src_2_out, dsize=out_size)

    # mapping src to out img
    img_warped_tgt = cv2.warpPerspective(img_tgt, homo_tgt_2_out, dsize=out_size)
    img_warped_tgt[img_warped > 0] = 0
    img_warped = img_warped + img_warped_tgt
    # mapping tgt to out img

    # compute overlapping
    img_blending, mask_overlapping = utils.img_blending.blend_by_median(img_tgt, img_warped[int(img_box_tgt_in_out[1]):int(img_box_tgt_in_out[3]), int(img_box_tgt_in_out[0]):int(img_box_tgt_in_out[2])])
    img_warped[int(img_box_tgt_in_out[1]):int(img_box_tgt_in_out[3]), int(img_box_tgt_in_out[0]):int(img_box_tgt_in_out[2])][mask_overlapping] = img_blending[mask_overlapping]

    cv2.namedWindow('src', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tgt', cv2.WINDOW_NORMAL)
    cv2.namedWindow('src_warp', cv2.WINDOW_NORMAL)
    cv2.imshow('src', img_src)
    cv2.imshow('tgt', img_tgt)
    cv2.imshow('src_warp', img_warped)
    cv2.waitKey(0)

    return img_warped, homo_tgt_2_out


def stitch_with_last_img():
    data_path = './data/head'
    img_paths = utils.dataset.get_iphone_data(data_path)
    print(img_paths)

    for i in range(len(img_paths)-1):
        img1, img2 = cv2.imread(img_paths[i]), cv2.imread(img_paths[i+1])
        homo_1_2_2 = utils.view_geometry.find_homo(img1, img2)

        img_src = img1 if i == 0 else img_tgt
        img_tgt = img2
        img_tgt, homo_tgt_2_out_new = warp_img(img_src, img_tgt, homo_1_2_2)


def stitch_with_first_img():
    data_path = './data/graf'
    img_paths, homo_paths = utils.dataset.get_robots_ox_ac_uk_vgg_data(data_path)
    for i in range(len(homo_paths)):
        img1_path, img2_path, homo_path = img_paths[0], img_paths[i+1], homo_paths[i]

        img1_org = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        f = open(homo_path)
        homo_1_2_2_dataset = np.loadtxt(homo_path)
        homo_1_2_2 = utils.view_geometry.find_homo(img1_org, img2)
        print('dataset', homo_1_2_2_dataset)
        print('compute', homo_1_2_2)

        img_src, img_tgt = img2, img1_org if i == 0 else img1
        homo_src_2_tgt = np.linalg.inv(homo_1_2_2_dataset)

        if i == 0:
            homo_tgt_2_out_last = np.eye(3)
        homo_src_2_tgt = np.matmul(homo_tgt_2_out_last, homo_src_2_tgt)

        img1, homo_tgt_2_out_new = warp_img(img_src, img_tgt, homo_src_2_tgt)
        homo_tgt_2_out_last = np.matmul(homo_tgt_2_out_new, homo_tgt_2_out_last)


def stitch_with_img_no_label():
    data_path = './data/head'
    img_paths = utils.dataset.get_iphone_data(data_path)
    for i in range(len(img_paths)):
        if i < 1:
            continue

        img1_path, img2_path = img_paths[i], img_paths[i+1]

        img1_org = cv2.imread(img1_path)

        img2 = cv2.imread(img2_path)
        homo_1_2_2 = utils.view_geometry.find_homo(img1_org, img2, flag_vis_feature_matching=True)

        print('compute', homo_1_2_2)

        img_src, img_tgt = img2, img1_org # if i == 0 else img1

        homo_src_2_tgt = np.linalg.inv(homo_1_2_2)

        if i == 1:
            homo_tgt_2_out_last = np.eye(3)
        homo_src_2_tgt = np.matmul(homo_tgt_2_out_last, homo_src_2_tgt)

        img1, homo_tgt_2_out_new = warp_img(img_src, img_tgt, homo_src_2_tgt)
        homo_tgt_2_out_last = np.matmul(homo_tgt_2_out_new, homo_tgt_2_out_last)


def main():
    # stitch_with_last_img()
    stitch_with_img_no_label()


if __name__ == '__main__':
    main()
