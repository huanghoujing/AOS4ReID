import numpy as np
import os.path as osp

ospj = osp.join
ospeu = osp.expanduser

from ..utils.utils import load_pickle
from .TrainSet import TrainSet
from .TestSet import TestSet


def create_dataset(
        name='market1501',
        part='trainval',
        **kwargs):
    assert name in ['market1501', 'cuhk03', 'duke'], \
        "Unsupported Dataset {}".format(name)

    ########################################
    # Specify Directory and Partition File #
    ########################################

    partition_file = 'dataset/{}/partitions.pkl'.format(name)
    assert part in ['trainval', 'test'], "Unsupported Dataset Part {} for Dataset {}".format(part, name)

    ##################
    # Create Dataset #
    ##################

    # Use standard Market1501 CMC settings for all datasets here.
    cmc_kwargs = dict(
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True
    )

    partitions = load_pickle(partition_file)

    if part == 'trainval':
        ret_set = TrainSet(
            im_names=partitions['{}_im_names'.format(part)],
            labels=partitions['trainval_labels'],
            **kwargs
        )

    elif part == 'test':
        kwargs.update(cmc_kwargs)
        ret_set = TestSet(
            im_names=partitions['{}_im_names'.format(part)],
            ids=partitions['test_ids'],
            cams=partitions['test_cams'],
            marks=partitions['test_marks'],
            **kwargs
        )

    if part == 'trainval':
        num_ids = len(set(partitions['trainval_labels']))
    else:
        num_ids = len(set(partitions['test_ids']))
        num_cams = len(set(partitions['test_cams']))
        num_query = np.sum(np.array(partitions['test_marks']) == 0)
        num_gallery = np.sum(np.array(partitions['test_marks']) == 1)

    # Print dataset information
    print('-' * 40)
    print('{} {} set'.format(name, part))
    print('-' * 40)
    print('NO. Images: {}'.format(len(partitions['{}_im_names'.format(part)])))
    print('NO. IDs: {}'.format(num_ids))

    try:
        print('NO. CAMs: {}'.format(num_cams))
        print('NO. Query Images: {}'.format(num_query))
        print('NO. Gallery Images: {}'.format(num_gallery))
    except:
        pass

    print('-' * 40)

    return ret_set
