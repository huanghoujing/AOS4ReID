"""To support the unified dataset interface, save following items:
- trainval_im_names
- trainval_labels
- test_im_names
- test_ids
- test_cams
- test_marks
"""

from __future__ import print_function

import sys

sys.path.insert(0, '.')

import os.path as osp
from PIL import Image
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm

from package.utils.utils import save_pickle
from package.utils.utils import may_make_dir
from package.utils.dataset_utils import get_im_names
from package.utils.dataset_utils import walkdir


def parse_original_im_name(im_name, parse_type='id'):
    """Get the person id or cam from an image name. Example: 0200_c2_1931.jpg"""
    assert parse_type in ('id', 'cam')
    im_name = osp.basename(im_name)
    if parse_type == 'id':
        parsed = int(im_name[:4])
    else:
        parsed = int(im_name[6])
    return parsed


def transform_trainval_set(im_dir, start_label=0):
    im_names = get_im_names(osp.join(im_dir, 'bounding_box_train'), pattern='*.jpg', return_path=True, return_np=False)
    im_names.sort()
    ids = [parse_original_im_name(n, 'id') for n in im_names]
    unique_ids = list(set(ids))
    unique_ids.sort()
    ids2labels = dict(zip(unique_ids, range(start_label, start_label + len(unique_ids))))
    labels = [ids2labels[id] for id in ids]
    return im_names, labels, len(unique_ids)


def transform_test_set(im_dir):
    # query
    q_im_names = get_im_names(osp.join(im_dir, 'query'), pattern='*.jpg', return_path=True, return_np=False)
    q_im_names.sort()
    q_ids = [parse_original_im_name(n, 'id') for n in q_im_names]
    q_cams = [parse_original_im_name(n, 'cam') for n in q_im_names]
    q_marks = [0 for _ in q_im_names]
    # gallery
    g_im_names = get_im_names(osp.join(im_dir, 'bounding_box_test'), pattern='*.jpg', return_path=True, return_np=False)
    g_im_names.sort()
    g_ids = [parse_original_im_name(n, 'id') for n in g_im_names]
    g_cams = [parse_original_im_name(n, 'cam') for n in g_im_names]
    g_marks = [1 for _ in g_im_names]
    #
    im_names = q_im_names + g_im_names
    ids = q_ids + g_ids
    cams = q_cams + g_cams
    marks = q_marks + g_marks

    return im_names, ids, cams, marks


def transform(im_dir, save_path):
    trainval_im_names, trainval_labels, _ = transform_trainval_set(im_dir)
    test_im_names, test_ids, test_cams, test_marks = transform_test_set(im_dir)

    partitions = {'trainval_im_names': trainval_im_names,
                  'trainval_labels': trainval_labels,
                  'test_im_names': test_im_names,
                  'test_ids': test_ids,
                  'test_cams': test_cams,
                  'test_marks': test_marks}
    save_pickle(partitions, save_path)


def save_png_as_jpg(cuhk03_dir, png_dir_name, jpg_dir_name):
    png_im_dir = osp.join(cuhk03_dir, png_dir_name)
    assert osp.exists(png_im_dir), "The PNG image dir {} should be place inside {}".format(png_dir_name, cuhk03_dir)
    # CUHK03 contains 14097 detected + 14096 labeled images = 28193
    num_ims = 0
    for png_path in tqdm(walkdir(png_im_dir, '.png'), desc='PNG->JPG', miniters=2000, ncols=120, unit=' images'):
        jpg_path = png_path.replace(png_dir_name, jpg_dir_name).replace('.png', '.jpg')
        may_make_dir(osp.dirname(jpg_path))
        imsave(jpg_path, np.array(Image.open(png_path)))
        num_ims += 1
    print('{} CUHK03-NP JPEG Images Saved to {}'.format(num_ims, osp.join(cuhk03_dir, jpg_dir_name)))


if __name__ == '__main__':
    cuhk03_dir = 'dataset/cuhk03'
    png_dir_name = 'cuhk03-np'
    jpg_dir_name = 'cuhk03-np-jpg'
    # save_png_as_jpg(cuhk03_dir, png_dir_name, jpg_dir_name)
    transform(osp.join(cuhk03_dir, jpg_dir_name, 'detected'), 'dataset/cuhk03/partitions.pkl')
