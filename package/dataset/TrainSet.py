from .Dataset import Dataset

from PIL import Image
import numpy as np


class TrainSet(Dataset):
    """Training set for identification loss.
    """

    def __init__(self,
                 im_names=None,
                 labels=None,
                 **kwargs):
        super(TrainSet, self).__init__(dataset_size=len(im_names), **kwargs)
        self.im_names = im_names
        self.labels = labels
        self.inds = range(len(im_names))

    def get_sample(self, ptr):
        """Get one sample to put to queue."""
        cur_ind = self.inds[ptr]
        im_name = self.im_names[cur_ind]
        im = np.asarray(Image.open(im_name))
        im, mirrored = self.pre_process_im(im)
        label = self.labels[cur_ind]
        return im, im_name, label, mirrored

    def next_batch(self):
        """Next batch of images and labels.
        Returns:
          ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
          im_names: a numpy array of image names, len(im_names) >= 1
          labels: a numpy array of image labels, len(labels) >= 1
          mirrored: a numpy array of booleans, whether the images are mirrored
          self.epoch_done: whether the epoch is over
        """
        if self.epoch_done and self.shuffle:
            self.prng.shuffle(self.inds)
        samples, self.epoch_done = self.prefetcher.next_batch()
        im_list, im_names, labels, mirrored = zip(*samples)
        # Transform the list into a numpy array with shape [N, ...]
        ims = np.stack(im_list, axis=0)
        im_names = np.array(im_names)
        labels = np.array(labels)
        mirrored = np.array(mirrored)
        return ims, im_names, labels, mirrored, self.epoch_done
