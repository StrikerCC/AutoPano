# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/2/21 5:46 PM
"""
import copy
import time
from itertools import compress

import cv2
import numpy as np
from scipy.spatial.kdtree import KDTree

# import read
import utils
import utils.vis


wait_key = 0


def make_chessbaord_corners_coord(chessboard_size, square_size):
    chessbaord_corners_coord = np.zeros((chessboard_size[0] * chessboard_size[1], 3))
    xy = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    chessbaord_corners_coord[:, :2] = xy
    return chessbaord_corners_coord * square_size


def get_checkboard_corners(img, checkboard_size, flag_vis=False):
    assert len(checkboard_size) == 2
    flag_found, corners = cv2.findChessboardCorners(img, checkboard_size)
    if flag_found:
        cv2.cornerSubPix(img, corners, winSize=(5, 5), zeroZone=(-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        if flag_vis:
            cv2.drawChessboardCorners(img, patternSize=checkboard_size, corners=corners, patternWasFound=True)
            cv2.imshow('checkboard corners', img)
            cv2.waitKey(0)
    return corners


def sift_features(img, flag_debug=False):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    if flag_debug:
        img_kp = cv2.drawKeypoints(img, kp, None)
        print(len(kp))
        print(kp[0])
        print(des.shape)
        cv2.namedWindow('kp', cv2.WINDOW_NORMAL)
        cv2.imshow('kp', img_kp)
        cv2.waitKey(wait_key)

    return kp, des


# def match_filter_pts_pair(kp1, des1, kp2, des2):
def match_filter_pts_pair(pts1, des1, pts2, des2):
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(des1, des2, k=2)

    '''filter with feature value ambiguity'''
    good_matches = [first for first, second in matches if first.distance < 0.85 * second.distance]
    index_match = np.asarray([[m.queryIdx, m.trainIdx] for m in good_matches])

    pts1 = pts1[index_match[:, 0].tolist()]     # queryIdx
    des1 = des1[index_match[:, 0]]

    pts2 = pts2[index_match[:, 1].tolist()]     # trainIdx
    des2 = des2[index_match[:, 1]]

    print('feature value ambiguity', len(good_matches), '/', len(matches), 'points left')

    '''filter with epi-polar geometry ransac'''
    F, mask = cv2.cv2.findFundamentalMat(pts1, pts2, cv2.cv2.FM_RANSAC, ransacReprojThreshold=2.0, confidence=0.9999,
                                         maxIters=10000)

    mask_ = mask.squeeze().astype(bool).tolist()

    index_match = index_match[mask_]
    index_match_set = set(index_match[:, 0].tolist())

    pts1 = pts1[mask_]
    des1 = des1[mask_]
    pts2 = pts2[mask_]
    des2 = des2[mask_]

    good_matches = [m for m in good_matches if m.queryIdx in index_match_set]

    print('epi-polar geometry ransac', len(good_matches), '/', len(matches), 'points left, epi-polar rms',
          np.mean(np.sum(np.matmul(np.hstack([pts1, np.ones((len(pts1), 1))]), F)*np.hstack([pts2, np.ones((len(pts1), 1))]), axis=1)))

    return pts1, des1, pts2, des2


def get_sift_and_pts(img1, img2, flag_debug=False):
    if img1 is None or img2 is None:
        return None

    shrink = 2.0
    img1_sub = cv2.resize(img1, (int(img1.shape[1] / shrink), int(img1.shape[0] / shrink)))
    img2_sub = cv2.resize(img2, (int(img2.shape[1] / shrink), int(img2.shape[0] / shrink)))

    # kp1, des1 = sift_features(img1_sub, flag_debug)
    # kp2, des2 = sift_features(img2_sub, flag_debug)

    kp1, des1 = sift_features(img1_sub)
    kp2, des2 = sift_features(img2_sub)

    pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 2) * shrink
    pts2 = np.float32([kp.pt for kp in kp2]).reshape(-1, 2) * shrink

    # pts1, des1, pts2, des2, good_matches = match_filter_pts_pair(kp1, des1, kp2, des2)
    pts1, des1, pts2, des2 = match_filter_pts_pair(pts1, des1, pts2, des2)

    '''statistics'''
    if flag_debug:
        print('Get ', len(pts1), ' good matches')
        img3 = utils.vis.draw_matches(img1, pts1, img2, pts2)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img3)
        cv2.waitKey(wait_key)

    return pts1, des1, pts2, des2


def get_pts_pair_by_sift(img1, img2, flag_debug=False):
    pts1, des1, pts2, des2, good_matches = get_sift_and_pts(img1, img2, flag_debug)
    return pts1, pts2


def get_pts(img):
    # TODO:
    pass


def main():
    return


if __name__ == '__main__':
    main()
