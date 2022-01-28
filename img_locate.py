import numpy
import cv2
import utils.dataset
import utils.view_geometry
import img_stitching


def main():
    mono_img_paths, stereo_img_paths = utils.dataset.get_triple_camera_data('./data/triple/')
    print(mono_img_paths)
    print(stereo_img_paths)

    for i_mono, img_mono_path in enumerate(mono_img_paths):
        img_mono = cv2.imread(img_mono_path)

        # get mono feature
        for i_stereo, (img_left_path, img_right_path) in enumerate(zip(stereo_img_paths['left'], stereo_img_paths['right'])):
            if i_mono not in {8, 9}:
                continue
            img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)

            for img_local in [img_left, img_right]:
                h, w = img_local.shape[0] / 4, img_local.shape[1] / 4

                # get local feature
                # match
                # find homograph
                homo = utils.view_geometry.find_homo(img_local, img_mono, flag_vis_feature_matching=True)
                if homo is None:
                    continue
                print(homo)

                # modify stereo img
                cv2.rectangle(img_local, (int(img_local.shape[1] / 2 - w), int(img_local.shape[0] / 2 - h)),
                              (int(img_local.shape[1] / 2 + w/2), int(img_local.shape[0] / 2 + h/2)), color=(0, 0, 255),
                              thickness=15)

                # mapping local img to global
                img_merge, homo_tgt_2_out_new = img_stitching.warp_img(img_local, img_mono, homo)

                # statistic
                print(i_mono, i_stereo)

                # vis
                cv2.namedWindow('mono', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('left', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('right', cv2.WINDOW_NORMAL)

                cv2.imshow('mono', img_merge)
                cv2.waitKey(1)

                # cv2.imshow('left', img_left)
                #
                # cv2.imshow('right', img_right)
                # cv2.waitKey(0)
            break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
