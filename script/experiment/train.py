from __future__ import print_function

import sys

sys.path.insert(0, '.')
print('sys.path is:\n\t{}'.format('\n\t'.join(sys.path)))

import torch
print('[PYTORCH VERSION]:', torch.__version__)
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import cv2

from package.dataset import create_dataset
from package.model.Model import Model

from package.utils.utils import time_str
from package.utils.utils import str2bool
from package.utils.utils import CommaSeparatedTuple
from package.utils.utils import load_state_dict
from package.utils.utils import load_ckpt
from package.utils.utils import save_ckpt
from package.utils.utils import set_devices
from package.utils.utils import AverageMeter
from package.utils.utils import to_scalar
from package.utils.utils import ReDirectSTD
from package.utils.utils import adjust_lr_staircase
from package.utils.utils import load_pickle


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='Baseline', choices=['Baseline', 'No-Adversary', 'Random', 'Hard-1', 'Sampling'])
    parser.add_argument('--sw_area', type=float, default=0.1, help='Sliding window area ratio')
    parser.add_argument('--sw_stride', type=int, default=10, help='Sliding window stride')
    parser.add_argument('--occlude_prob', type=float, default=0.5, help='Occlusion probability in tasks except Baseline')

    parser.add_argument('-d', '--sys_device_ids', type=CommaSeparatedTuple(func=int), default=(0,))
    parser.add_argument('--train_set', type=str, default='market1501', choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--test_sets', type=CommaSeparatedTuple(func=str), default=('market1501',))

    parser.add_argument('--num_prefetch_threads', type=int, default=2, help="NO. threads for prefetching data")
    parser.add_argument('--prefetch_size', type=int, default=200, help="Queue size for prefetching data")

    # Cropping is not used in this project.
    # Horizontal Flipping is the only data augmentation in training.
    # No augmentation is used for testing.
    parser.add_argument('--crop_prob', type=float, default=0, help="The probability of each image to go through cropping.")
    parser.add_argument('--crop_ratio', type=float, default=1, help="Image area ratio to crop.")
    parser.add_argument('--resize_h_w', type=CommaSeparatedTuple(func=int), default=(256, 128), help="The final image size for network input, for both training and testing.")

    parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1, 2], help="Stride of ResNet's last stage.")
    parser.add_argument('--max_or_avg', type=str, default='avg', choices=['avg', 'max'], help="Global pooling after ResNet's last stage.")
    parser.add_argument('--dropout_rate', type=float, default=None, help="You can set some value to overwrite the default that is defined in the later code.")

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)

    parser.add_argument('--finetuned_params_lr', type=float, default=0.01)
    parser.add_argument('--new_params_lr', type=float, default=0.02)

    parser.add_argument('--staircase_decay_at_epochs', type=CommaSeparatedTuple(func=int), default=(26, 51))
    parser.add_argument('--staircase_decay_multiply_factor', type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=60)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--steps_per_log', type=int, default=50)
    parser.add_argument('--epochs_per_val', type=int, default=10)

    args = parser.parse_args()
    return args


def join(sequence, sep):
    sequence = [s for s in sequence if s != '']
    return sep.join(sequence)


def params_to_mask(im_h, im_w, ch, cw, h, w, fractional=False):
    """Transform parameters to a mask.
    Args:
        im_h: height of image, integer
        im_w: width of image, integer
        ch: center coordinate along the height axis, integer or 0-1 float
        cw: center coordinate along the width axis, integer or 0-1 float
        h: height of zeros region of the mask, integer
        w: width of zeros region of the mask, integer
        fractional: When True, `cw, ch, h, w` should be in range (0, 1).
    Returns:
        mask: a numpy array with shape [im_h, im_w]
    """
    mask = np.ones([im_h, im_w])
    h_mul = im_h if fractional else 1
    w_mul = im_w if fractional else 1
    h1 = max(int((ch - 0.5 * h) * h_mul), 0)
    h2 = min(int((ch + 0.5 * h) * h_mul), im_h)
    w1 = max(int((cw - 0.5 * w) * w_mul), 0)
    w2 = min(int((cw + 0.5 * w) * w_mul), im_w)
    mask[h1:h2, w1:w2] = 0
    return mask


def gen_rand_mask(
        im_h, im_w,
        area_ratio_low, area_ratio_high,
        aspect_ratio_low, aspect_ratio_high,
):
    """Generate a random rectangular mask placed at some position of the image.
    Args:
        im_h: height of image, integer
        im_w: width of image, integer
        area_ratio_low: in range (0, 1)
        area_ratio_high: in range (0, 1)
        aspect_ratio_low: the lower bound of `w / h`
        aspect_ratio_high: the higher bound of `w / h`
    Returns:
        mask: a numpy array with shape [im_h, im_w]
    """
    area_ratio = np.random.uniform(area_ratio_low, area_ratio_high)
    area = im_w * im_h * area_ratio

    success = False
    for i in range(10):
        # aspect = w / h
        aspect = np.random.uniform(aspect_ratio_low, aspect_ratio_high)
        w = int(np.sqrt(area * aspect))
        h = int(np.sqrt(area / float(aspect)))
        if (w < im_w) and (h < im_h):
            success = True
            # print('Gen random mask successfully after {} trial, h = {}, w = {}'
            #       .format(i + 1, h, w))
            break
    if not success:
        aspect = float(im_w) / im_h
        w = int(np.sqrt(area * aspect))
        h = int(np.sqrt(area / float(aspect)))
    cw = int(np.random.uniform(w * 0.5, im_w - w * 0.5))
    ch = int(np.random.uniform(h * 0.5, im_h - h * 0.5))
    mask_params = (im_h, im_w, ch, cw, h, w)
    mask = params_to_mask(*mask_params)
    return mask


def blur_prob_diff(prob_diff):
    """Blur each prob_diff map (with shape [num_h_pos, num_w_pos]) using a 3x3 kernel.
    Whether it's effective is not analysed -- it's just intuitively applied."""
    return {m_key: {im_name: cv2.blur(p_d_, (3, 3))
                    for im_name, p_d_ in p_d.items()}
            for m_key, p_d in prob_diff.items()}


def get_masks(im_names, mirrored, cfg, all_masks=None, prob_diff=None):
    """Get a batch of masks for the input batch.
    Returns:
        masks: numpy array with shape [len(im_names), im_h, im_w]
    """
    masks = []
    for name, m in zip(im_names, mirrored):
        if np.random.uniform() < cfg.occlude_prob:
            if cfg.task == 'Random':
                mask = gen_rand_mask(cfg.resize_h_w[0], cfg.resize_h_w[1], cfg.sw_area, cfg.sw_area, 1, 1)
            else:
                m_key = 'mirrored' if m else 'non_mirrored'
                p_d = prob_diff[m_key][name]
                p_d = p_d.flatten()
                if cfg.task == 'No-Adversary':
                    ind = np.argsort(p_d)[0]
                elif cfg.task == 'Hard-1':
                    ind = np.argsort(- p_d)[0]
                else:  # Sampling
                    p_d[p_d < 0] = 0
                    p = p_d / np.sum(p_d)
                    ind = np.random.choice(range(len(p_d)), p=p)
                mask = all_masks[ind]
        else:
            mask = np.ones(cfg.resize_h_w)
        masks.append(mask)
    masks = np.stack(masks)
    return masks


def main():
    cfg = parse_args()

    exp_dir = 'exp/{}_train_{}'.format(cfg.train_set, cfg.task)
    # Redirect logs to both console and file.
    ReDirectSTD(osp.join(exp_dir, 'stdout_{}.txt'.format(time_str())), 'stdout', False)
    ReDirectSTD(osp.join(exp_dir, 'stderr_{}.txt'.format(time_str())), 'stderr', False)
    ckpt_file = osp.join(exp_dir, 'ckpt.pth')
    model_weight_file = osp.join(exp_dir, 'model_weight.pth')
    writer = SummaryWriter(log_dir=osp.join(exp_dir, 'tensorboard'))

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

    train_set_kwargs = dict(
        name=cfg.train_set,
        part='trainval',
        batch_size=cfg.train_batch_size,
        final_batch=False,
        shuffle=True,
        crop_prob=cfg.crop_prob,
        crop_ratio=cfg.crop_ratio,
        mirror_type='random',
        prng=np.random,
    )

    test_set_kwargs = dict(
        part='test',
        batch_size=cfg.test_batch_size,
        final_batch=True,
        shuffle=False,
        mirror_type=None,
        prng=np.random,
    )

    train_set_kwargs.update(dataset_kwargs)
    train_set = create_dataset(**train_set_kwargs)

    test_set_kwargs.update(dataset_kwargs)
    test_sets = []
    for name in cfg.test_sets:
        test_set_kwargs['name'] = name
        test_sets.append(create_dataset(**test_set_kwargs))

    ###########
    # Models  #
    ###########

    TVT, TMO = set_devices(cfg.sys_device_ids)

    # You may find that dropout=0 is also OK under current lr settings.
    if cfg.dropout_rate is not None:
        dropout_rate = cfg.dropout_rate
    elif cfg.train_set == 'market1501':
        dropout_rate = 0.6
    else:
        dropout_rate = 0.5
    model = Model(
        last_conv_stride=cfg.last_conv_stride,
        max_or_avg=cfg.max_or_avg,
        dropout_rate=dropout_rate,
        num_classes=len(set(train_set.labels)),
    )
    # Model wrapper
    model_w = DataParallel(model)

    #############################
    # Criteria and Optimizers   #
    #############################

    criterion = torch.nn.CrossEntropyLoss()

    # To finetune from ImageNet weights
    finetuned_params = list(model.base.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters() if not n.startswith('base.')]
    param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr},
                    {'params': new_params, 'lr': cfg.new_params_lr}]
    optimizer = optim.SGD(
        param_groups,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Bind them together just to save some codes in the following usage.
    modules_optims = [model, optimizer]

    ################################
    # May Resume Models and Optims #
    ################################

    if cfg.resume:
        resume_ep, scores = load_ckpt(modules_optims, ckpt_file)

    # May Transfer Models and Optims to Specified Device. Transferring optimizer
    # is to cope with the case when you load the checkpoint to a new device.
    TMO(modules_optims)

    ########
    # Test #
    ########

    def extract_feat(ims):
        model.eval()
        ims = Variable(TVT(torch.from_numpy(ims).float()))
        feat, logits = model_w(ims)
        feat = feat.data.cpu().numpy()
        return feat

    def test(load_model_weight=False):
        if load_model_weight:
            if model_weight_file != '':
                sd = torch.load(model_weight_file, map_location=(lambda storage, loc: storage))
                load_state_dict(model, sd)
                print('Loaded model weights from {}'.format(model_weight_file))
            else:
                load_ckpt(modules_optims, ckpt_file)

        for test_set, name in zip(test_sets, cfg.test_sets):
            if test_set.extract_feat_func is None:
                test_set.set_feat_func(extract_feat)
            print('\n=========> Test on dataset: {} <=========\n'.format(name))
            test_set.eval(
                normalize_feat=True,
                to_re_rank=False,
                verbose=False,
            )

    if cfg.only_test:
        test(load_model_weight=True)
        return

    ############
    # Training #
    ############

    prob_diff, all_masks = None, None
    if cfg.task in ['No-Adversary', 'Hard-1', 'Sampling']:
        prob_diff = load_pickle('exp/{}_sw_occlusion/prob_diff.pkl'.format(cfg.train_set))
        prob_diff = blur_prob_diff(prob_diff)
        all_masks = load_pickle('exp/{}_sw_occlusion/all_masks.pkl'.format(cfg.train_set))

    start_ep = resume_ep if cfg.resume else 0
    for ep in range(start_ep, cfg.total_epochs):

        # Adjust Learning Rate
        adjust_lr_staircase(
            optimizer.param_groups,
            [cfg.finetuned_params_lr, cfg.new_params_lr],
            ep + 1,
            cfg.staircase_decay_at_epochs,
            cfg.staircase_decay_multiply_factor,
        )

        model.train()

        # For recording loss
        loss_meter = AverageMeter(name='cls loss')

        ep_st = time.time()
        step = 0
        epoch_done = False
        while not epoch_done:

            step += 1
            step_st = time.time()

            ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

            # Occlude images before feeding to network
            if cfg.task != 'Baseline':
                masks = get_masks(im_names, mirrored, cfg, all_masks=all_masks, prob_diff=prob_diff)
                ims = ims * np.expand_dims(masks, 1)

            ims_var = Variable(TVT(torch.from_numpy(ims).float()))
            labels_var = Variable(TVT(torch.from_numpy(labels).long()))

            _, logits = model_w(ims_var)
            loss = criterion(logits, labels_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(to_scalar(loss))

            if step % cfg.steps_per_log == 0:
                time_log = '\tStep {}/Ep {}, {:.2f}s'.format(step, ep + 1, time.time() - step_st, )
                loss_log = loss_meter.val_str
                log = join([time_log, loss_log], ', ')
                print(log)

        #############
        # Epoch Log #
        #############

        time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st, )
        loss_log = loss_meter.avg_str
        log = join([time_log, loss_log], ', ')
        print(log)

        writer.add_scalars(
            loss_meter.name,
            {loss_meter.name: loss_meter.avg},
            ep,
        )

        ########
        # Test #
        ########

        if ((ep + 1) % cfg.epochs_per_val == 0) or ((ep + 1) == cfg.total_epochs):
            test(load_model_weight=False)

        #############
        # Save CKPT #
        #############

        save_ckpt(modules_optims, ep + 1, 0, ckpt_file)


if __name__ == '__main__':
    main()
