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

from package.utils.utils import save_pickle
from package.utils.dataset_utils import get_im_names


def parse_original_im_name(im_name, parse_type='id'):
    """Get the person id or cam from an image name."""
    assert parse_type in ('id', 'cam')
    im_name = osp.basename(im_name)
    if parse_type == 'id':
        parsed = int(im_name[:4])
    else:
        parsed = int(im_name[6])
    return parsed


def transform_trainval_set(im_dir, start_label=0):
    im_names = get_im_names(osp.join(im_dir, 'bounding_box_train'), return_path=True, return_np=False)
    im_names.sort()
    ids = [parse_original_im_name(n, 'id') for n in im_names]
    unique_ids = list(set(ids))
    unique_ids.sort()
    ids2labels = dict(zip(unique_ids, range(start_label, start_label + len(unique_ids))))
    labels = [ids2labels[id] for id in ids]
    return im_names, labels, len(unique_ids)


def transform_test_set(im_dir):
    # query
    q_im_names = get_im_names(osp.join(im_dir, 'query'), return_path=True, return_np=False)
    q_im_names.sort()
    q_ids = [parse_original_im_name(n, 'id') for n in q_im_names]
    q_cams = [parse_original_im_name(n, 'cam') for n in q_im_names]
    q_marks = [0 for _ in q_im_names]
    # gallery
    g_im_names = get_im_names(osp.join(im_dir, 'bounding_box_test'), return_path=True, return_np=False)
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


if __name__ == '__main__':
    im_dir = 'dataset/duke/DukeMTMC-reID'
    save_path = 'dataset/duke/partitions.pkl'
    transform(im_dir, save_path)
