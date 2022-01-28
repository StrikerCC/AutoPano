
import time

import cv2
import numpy as np
import open3d as o3
import transforms3d as tf3

import camera.calibration
import utils.dataset
import utils.feature
import utils.mapping
from camera.cam import BiCamera
from camera.calibration import stereo_calibrate
import utils.vis


def main():
    """3d reconstruction"""
    dataset_dir = '/home/cheng/Pictures/data/202201251506/'
    data_stereo = utils.dataset.get_calibration_and_img(dataset_dir)
    data_stereo['left_general_img'] = data_stereo['left_general_img'][4:]
    data_stereo['right_general_img'] = data_stereo['right_general_img'][4:]

    for key in data_stereo.keys():
        print(key, 'has')
        for img_dir in data_stereo.get(key):
            print('     ', img_dir)

    stereo = BiCamera('./config/bicam_cal_para.json')

    # square_size = 3.0
    # checkboard_size = (11, 8)  # (board_width, board_height)
    # binocular = BiCamera()
    # camera.calibration.stereo_calibrate(square_size, checkboard_size, data_binocular['left_calibration_img'], data_binocular['right_calibration_img'], binocular=binocular,
    #                  file_path_2_save='./config/bicam_cal_para_.json')

    print('calibration result')
    print(stereo.cam_left.camera_matrix)
    print(stereo.cam_right.camera_matrix)

    # 3d recon
    last_frame = {'pts_3d': None,
                  'pts_color': None,
                  'pts_2d_left': None,
                  'pts_2d_right': None,
                  'sift_left': None,
                  'sift_right': None,
                  'match': None}

    pc_general = o3.geometry.PointCloud()

    for i_view, (img_left_path, img_right_path) in enumerate(zip(data_stereo['left_general_img'], data_stereo['right_general_img'])):
        print('running stereo vision on ', i_view, img_left_path, img_right_path)

        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)

        gray_left, gray_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # extract feature
        time_start = time.time()
        pts_2d_left, sift_left, pts_2d_right, sift_right = utils.feature.get_sift_and_pts(gray_left, gray_right, flag_debug=True)

        print('get features in', time.time() - time_start, 'seconds')

        # compute feature 3d coord
        pts_3d_general_skin_in_left = stereo.transform_raw_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)

        # get color for 3d feature points
        color_general_skin = (img_left[pts_2d_left.T[::-1].astype(int).tolist()] + img_right[pts_2d_right.T[::-1].astype(int).tolist()])[::-1] / 2 / 255.0   # take average of left and right bgr, then convert to normalized rgb

        # get triangle mesh for mesh body

        '''mapping to global camera frame'''
        # compute interested 3d pts in global frame
        tf_left_2_right = utils.mapping.rt_2_tf(stereo.r, stereo.t)
        # pts_3d_general_skin_in_right = utils.mapping.transform_pt_3d(tf=tf_left_2_right, pts=pts_3d_general_skin_in_left)
        pts_3d_general_skin_in_right = pts_3d_general_skin_in_left

        # compute 2d pixel coord of interested 3d pts in global pixel frame

        # pts_2d_general_skin_in_right = stereo.cam_right.proj_and_distort(pts_3d_general_skin_in_right) # projection
        pts_2d_general_skin_in_right, *_ = cv2.projectPoints(
            pts_3d_general_skin_in_right,
            rvec=cv2.Rodrigues(stereo.r)[0], tvec=stereo.t,
            cameraMatrix=stereo.cam_right.camera_matrix,
            distCoeffs=stereo.cam_right.distortion_coefficient)

        # vis skin points in local and global
        for pts_2d in pts_2d_general_skin_in_right[:, 0, :]:
            pts_2d = pts_2d.astype(int)
            img_right = cv2.circle(img_right, center=pts_2d, radius=15, color=(0, 0, 255), thickness=5)

        # statistics
        # vis
        cv2.namedWindow('hair skin pt in global img', cv2.WINDOW_NORMAL)
        cv2.imshow('hair skin pt in global img', img_right)
        cv2.waitKey(0)

        '''vis left main'''
        distance_max = 500
        mesh = o3.geometry.TriangleMesh()
        frame_left = mesh.create_coordinate_frame(size=25.0)
        frame_right = mesh.create_coordinate_frame(size=45.0)
        frame_right.transform(tf_left_2_right)

        # mask out some wrong point by distance away from camera
        # pts_3d = pts_3d[np.linalg.norm(pts_3d, axis=-1) < distance_max]
        pc_general.points = o3.utility.Vector3dVector(pts_3d_general_skin_in_left)
        pc_general.colors = o3.utility.Vector3dVector(color_general_skin)

        o3.visualization.draw_geometries([pc_general, frame_left, frame_right])
        o3.visualization.draw_geometries([pc_general])

        '''vis right main'''
        frame_left = mesh.create_coordinate_frame(size=25.0)
        frame_left.transform(np.linalg.inv(tf_left_2_right))
        frame_right = mesh.create_coordinate_frame(size=45.0)

        # mask out some wrong point by distance away from camera
        # pts_3d = pts_3d[np.linalg.norm(pts_3d, axis=-1) < distance_max]
        pc_general.points = o3.utility.Vector3dVector(pts_3d_general_skin_in_right)
        pc_general.colors = o3.utility.Vector3dVector(color_general_skin)

        o3.visualization.draw_geometries([pc_general, frame_left, frame_right])
        o3.visualization.draw_geometries([pc_general])


if __name__ == '__main__':
    main()
