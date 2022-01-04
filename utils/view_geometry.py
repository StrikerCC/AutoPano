# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/10/21 3:58 PM
"""
import cv2
import utils.match_by_feature


def find_homo(img_src, img_tgt, flag_vis_feature_matching=False):
    pts_src, pts_tgt = utils.match_by_feature.match_pts(img_src, img_tgt, flag_vis_feature_matching)
    assert pts_src.shape[0] > 4 and pts_tgt.shape[0] > 4, 'Not enough points ' + str(pts_src.shape[0]) + ' and ' + str(pts_tgt.shape[0])
    homo, status = cv2.findHomography(pts_src, pts_tgt, method=cv2.RANSAC, ransacReprojThreshold=4)
    return homo
