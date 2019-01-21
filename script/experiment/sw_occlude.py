from __future__ import print_function
import sys
sys.path.insert(0, '.')
print('sys.path is:\n\t{}'.format('\n\t'.join(sys.path)))

import torch
print('[PYTORCH VERSION]:', torch.__version__)
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import os.path as osp
import time
import numpy as np

from package.dataset import create_dataset
from package.model.Model import Model

from package.utils.utils import time_str
from package.utils.utils import ReDirectSTD
from package.utils.utils import set_devices
from package.utils.utils import save_pickle

# Since arguments are already defined in `train.py`, we reuse this `parse_args` function.
# Some arguments in `parse_args` are not used in this script, which does not matter.
from train import parse_args


def get_sw_positions(im_h_w=(256, 128), sw_h_w=(57, 57), stride=10):
    """Get all possible top-left positions of the given sliding window."""
    h_pos = range(0, im_h_w[0] - sw_h_w[0], stride)
    w_pos = range(0, im_h_w[1] - sw_h_w[1], stride)
    return h_pos, w_pos


def gen_masks(im_h_w=(256, 128), sw_h_w=(57, 57), stride=10):
    """Generate masks with zero-value rectangles at different positions of the mask.
    Returns:
        masks: numpy array with shape [num_possible_positions, im_h, im_w]
    """
    masks = []
    h_pos, w_pos = get_sw_positions(im_h_w, sw_h_w, stride)
    for h in h_pos:
        for w in w_pos:
            mask = np.ones(shape=im_h_w)
            mask[h:h + sw_h_w[0], w:w + sw_h_w[1]] = 0
            masks.append(mask)
    masks = np.stack(masks)
    return masks


def main():
    cfg = parse_args()

    ckpt_file = 'exp/{}_train_Baseline/ckpt.pth'.format(cfg.train_set)
    exp_dir = 'exp/{}_sw_occlusion'.format(cfg.train_set)
    # Redirect logs to both console and file.
    ReDirectSTD(osp.join(exp_dir, 'stdout_{}.txt'.format(time_str())), 'stdout', False)
    ReDirectSTD(osp.join(exp_dir, 'stderr_{}.txt'.format(time_str())), 'stderr', False)

    TVT, TMO = set_devices(cfg.sys_device_ids)

    # Dump the configurations to log.
    import pprint
    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)

    ###########
    # Dataset #
    ###########

    im_mean = [0.486, 0.459, 0.408]
    im_std = [0.229, 0.224, 0.225]

    dataset_kwargs = dict(
        resize_h_w=cfg.resize_h_w,
        scale=True,
        im_mean=im_mean,
        im_std=im_std,
        batch_dims='NCHW',
        num_prefetch_threads=cfg.num_prefetch_threads,
        prefetch_size=cfg.prefetch_size,
    )

    # batch_size=1, final_batch=True, mirror_type=None
    train_set_kwargs = dict(
        name=cfg.train_set,
        part='trainval',
        batch_size=1,
        final_batch=True,
        shuffle=True,
        crop_prob=cfg.crop_prob,
        crop_ratio=cfg.crop_ratio,
        mirror_type=None,
        prng=np.random,
    )

    train_set_kwargs.update(dataset_kwargs)
    train_set = create_dataset(**train_set_kwargs)

    #########
    # Model #
    #########

    model = Model(
        last_conv_stride=cfg.last_conv_stride,
        max_or_avg=cfg.max_or_avg,
        num_classes=len(set(train_set.labels)),
    )
    # Model wrapper
    model_w = DataParallel(model)

    ckpt = torch.load(ckpt_file, map_location=(lambda storage, loc: storage))
    model.load_state_dict(ckpt['state_dicts'][0])
    print('Loaded model weights from {}'.format(ckpt_file))
    model.eval()
    # Transfer Models to Specified Device
    TMO([model])

    ############################
    # Sliding Window Occlusion #
    ############################

    l = int(np.sqrt(cfg.sw_area * np.prod(cfg.resize_h_w)))
    sw_h_w = [l, l]
    all_masks = gen_masks(cfg.resize_h_w, sw_h_w, cfg.sw_stride)
    h_pos, w_pos = get_sw_positions(cfg.resize_h_w, sw_h_w, cfg.sw_stride)
    print('Num of all possible masks: {} * {} = {}'.format(len(h_pos), len(w_pos), len(all_masks)))

    def sw_occlude():
        """Calculate the probability difference caused by occluding different positions."""
        im_names, prob_diff = [], []
        epoch_done = False
        num_ims = 0
        st_time = time.time()
        last_time = time.time()
        # For each image
        while not epoch_done:
            num_ims += 1

            im, im_name, label, _, epoch_done = train_set.next_batch()
            im_names.append(im_name[0])

            im = Variable(TVT(torch.from_numpy(im).float()))
            label = label[0]

            # Calculate the original prob.
            # feat, logits = model_w(im)
            # ori_prob = F.softmax(logits, 1).data.cpu().numpy()[0, label]

            # To save time, here just use 1.
            ori_prob = 1

            probs = []
            # In order not to over flood the GPU memory, split into small batches.
            for masks in np.array_split(all_masks, int(len(all_masks) / 32) + 1):
                # Repeat an image for num_masks times.
                # `im` with shape [1, C, H, W] => `repeated_im` with shape [num_masks, C, H, W]
                repeated_im = im.repeat(len(masks), 1, 1, 1)
                # Repeat each mask for C times.
                # `masks` shape [num_masks, H, W] => [num_masks, C, H, W]
                masks = Variable(TVT(torch.from_numpy(masks).float().unsqueeze(1).expand_as(repeated_im)))
                # `logits` with shape [num_masks, num_classes]
                feat, logits = model_w(repeated_im * masks)
                probs_ = F.softmax(logits, 1)
                probs_ = probs_.data.cpu().numpy()[:, label].flatten()
                probs.append(probs_)
            # with shape [num_h_pos, num_w_pos], it can be resized to im shape for visualization
            probs = np.reshape(np.concatenate(probs), [len(h_pos), len(w_pos)])
            prob_diff.append(ori_prob - probs)

            if num_ims % 50 == 0:
                print('\t{}/{} images done, +{:.2f}s, total {:.2f}s'.format(num_ims, len(train_set.im_names), time.time() - last_time, time.time() - st_time))
                last_time = time.time()

        prob_diff = dict(zip(im_names, prob_diff))
        return prob_diff

    print('Sliding Window Occlusion for Non-mirrored Images.')
    train_set.set_mirror_type(None)
    prob_diff = dict()
    prob_diff['non_mirrored'] = sw_occlude()

    print('Sliding Window Occlusion for Mirrored Images.')
    train_set.set_mirror_type('always')
    prob_diff['mirrored'] = sw_occlude()

    save_pickle(prob_diff, osp.join(exp_dir, 'prob_diff.pkl'))
    save_pickle(all_masks, osp.join(exp_dir, 'all_masks.pkl'))


if __name__ == '__main__':
    main()
