"""

"""

import numpy as np
import os
import cv2

from PIL import Image
import scipy.ndimage
import scipy.misc
import shutil

RGB_SCALE = np.array([256, 256], dtype=np.float32)
GRAY_SCALE = np.array([256, 256, 3], dtype=np.float32)


def mkval(root_dir, subfolder):
    """
    Args:
      root_dir: '/home/thinkpad/Downloads/DT/pix2pix_data/stimuli'
      subfolder: 'text'
    Return:
    """
    out_dir = os.path.join(root_dir, 'val')
    print('out_dir: {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    input_dir = os.path.join(root_dir, subfolder)
    print('input_dir: {}'.format(input_dir))
    names_A = os.listdir(input_dir)
    print(len(names_A))

    val = np.random.choice(len(names_A), 10, replace=False)
    print(len(val))
    for i in val:
        shutil.move(os.path.join(input_dir, names_A[i]), os.path.join(out_dir, names_A[i]))


def rename_files(base_dir, mode):
    """
    Args:
      base_dir: '/home/thinkpad/Downloads/DT/pix2pix_data'
      mode: 'train', 'val'
    Return:
    """

    A_dir = os.path.join(base_dir, 'A', mode)
    B_dir = os.path.join(base_dir, 'B', mode)
    C_dir = os.path.join(base_dir, 'C', mode)
    names_A = os.listdir(A_dir)
    names_B = os.listdir(B_dir)
    names_C = os.listdir(C_dir)
    print(len(names_A))
    print(len(names_B))
    print(len(names_C))

    assert set(names_A) == set(names_B), "names_A != names_B"
    assert set(names_A) == set(names_C), "names_A != names_C"

    i = 1101
    for name in names_A:
        new_name = str(i) + '.png'
        os.rename(os.path.join(A_dir, name),
                  os.path.join(A_dir, new_name))
        os.rename(os.path.join(B_dir, name),
                  os.path.join(B_dir, new_name))
        os.rename(os.path.join(C_dir, name),
                  os.path.join(C_dir, new_name))
        i += 1


def load_display():
    names_train = os.listdir('./facades_renamed/train/B')
    for name in names_train:
        img = cv2.imread(os.path.join('./facades_renamed/train/B', name))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join('./facades_renamed/train/B_grayscale', name), img)

    names_val = os.listdir('./facades_renamed/val/B')
    for name in names_val:
        img = cv2.imread(os.path.join('./facades_renamed/val/B', name))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join('./facades_renamed/val/B_grayscale', name), img)


def resize_img(root_dir, subfolder):
    """
    Args:
      root_dir: '/home/thinkpad/Downloads/DT/pix2pix_data/A'
      subfolder: 'train', 'val'
    Return:
    """
    output_dir = os.path.join(root_dir, '{0}_resized'.format(subfolder))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_dir = os.path.join(root_dir, subfolder)
    imgs = os.listdir(input_dir)
    print(imgs)
    for img in imgs:
        # try:
        im = np.array(Image.open(os.path.join(input_dir, img)), dtype=np.float32)
        # print(im.shape)
        # coarse_img = \
        #     scipy.ndimage.interpolation.zoom(
        #         im,
        #         tuple(RGB_SCALE / np.asarray(im.shape, dtype=np.float32)),
        #         np.dtype(np.float32),
        #         mode='nearest'
        #     )
        coarse_img = cv2.resize(im, tuple(RGB_SCALE), interpolation=cv2.INTER_NEAREST)

        prefix = os.path.splitext(img)[0] + '.png'
        cv2.imwrite(os.path.join(output_dir, prefix), coarse_img)
        # except:
        #     print(os.path.join(dir, 'test', img))


def edge_detection(root_dir, subfolder):
    output_dir = os.path.join(root_dir, '{0}_50, 100'.format(subfolder))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    imgs = os.listdir(os.path.join(root_dir, subfolder))
    print(imgs)
    for img in imgs:
        im = cv2.imread(os.path.join(root_dir, subfolder, img), 0)
        # edges = cv2.Canny(im, 100, 200)
        edges = cv2.Canny(im, 50, 100)  # more detailed than above line.

        prefix = os.path.splitext(img)[0] + '.png'
        cv2.imwrite(os.path.join(output_dir, prefix), edges)


def msers_detection(root_dir, subfolder):
    output_dir = os.path.join(root_dir, '{0}_msers'.format(subfolder))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    imgs = os.listdir(os.path.join(root_dir, subfolder))
    print(imgs)

    # for img in imgs:
    #     im = cv2.imread(os.path.join(root_dir, subfolder, img), cv2.IMREAD_GRAYSCALE)
    #     mser = cv2.MSER()
    #     mser_areas = mser.detect(im)
    #
    #     prefix = os.path.splitext(img)[0] + '.png'
    #     cv2.imwrite(os.path.join(output_dir, prefix), edges)


# def feat_extr(root_dir, subfolder):
#     output_dir = os.path.join(root_dir, '{0}_SF'.format(subfolder))
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#     input_dir = os.path.join(root_dir, subfolder)
#     imgs = os.listdir(input_dir)
#     print(imgs)
#
#     img = cv2.imread(os.path.join(input_dir, imgs[0]), cv2.IMREAD_GRAYSCALE)
#     # print("----")
#     # print(img.shape)
#     # print(img.dtype)
#     # print(img)
#
#     # Create MSER object
#     mser = cv2.MSER_create()
#
#     # Your image path i-e receipt path
#     # img = cv2.imread('/home/rafiullah/PycharmProjects/python-ocr-master/receipts/73.jpg')
#
#     # Convert to gray scale
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     vis = img.copy()
#
#     # detect regions in gray scale image
#     regions, _ = mser.detectRegions(img)
#
#     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#
#     cv2.polylines(vis, hulls, 1, (0, 255, 0))
#     cv2.imshow('img', vis)
#     cv2.waitKey(0)
#
#     mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
#
#     for contour in hulls:
#         cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
#
#     # # this is used to find only text regions, remaining are ignored
#     # text_only = cv2.bitwise_and(img, img, mask=mask)
#     # cv2.imshow("text only", text_only)
#     # cv2.waitKey(0)


if __name__ == '__main__':
    # mkval('/home/thinkpad/Downloads/DT/pix2pix_data/stimuli', 'mixed')
    # rename_files('/home/thinkpad/Downloads/DT/pix2pix_data/', 'val')
    resize_img('/home/thinkpad/Downloads/DT/webpage_data/pix2pix_data/B', 'val')
    # feat_extr('/home/thinkpad/Downloads/DT/webpage_data/pix2pix_data/A', 'test')
    # edge_detection('/home/thinkpad/Downloads/DT/webpage_data/pix2pix_data/A', 'test')
    # msers_detection('/home/thinkpad/Downloads/DT/webpage_data/pix2pix_data/A', 'test')
