# About

This project implements paper [Adversarially Occluded Samples for Person Re-identification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Adversarially_Occluded_Samples_CVPR_2018_paper.pdf) using [pytorch](https://github.com/pytorch/pytorch).


# Requirements

- Python 2.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/) for easy package management.)
- [Pytorch 0.3](http://pytorch.org/)

The other packages and versions are listed in `requirements.txt`. You can install them by `pip install -r requirements.txt`.

# Prepare Datasets

Create directory `dataset` under the project directory, then place datasets in it, as follows.

```
${project_dir}/dataset
    market1501
        Market-1501-v15.09.15                   # Extracted from Market-1501-v15.09.15.zip, http://www.liangzheng.org/Project/project_reid.html
    cuhk03
        cuhk03-np                               # Extracted from cuhk03-np.zip, https://pan.baidu.com/s/1RNvebTccjmmj1ig-LVjw7A
    duke
        DukeMTMC-reID                           # Extracted from DukeMTMC-reID.zip, https://github.com/layumi/DukeMTMC-reID_evaluation
```

Then run following command to transform datasets.

```bash
python script/dataset/transform_market1501.py
python script/dataset/transform_cuhk03.py
python script/dataset/transform_duke.py
```

# Experiments

## Step I: Train Baseline

To train `Baseline` on Market1501, with GPU 0, run

```bash
bash script/experiment/train.sh Baseline market1501 0
```

## Step II: Sliding Window Occlusion

To apply sliding window occlusion with the trained `Baseline` model and obtain recognition probability, for Market1501, with GPU 0, run

```bash
bash script/experiment/sw_occlude.sh market1501 0
```

## Step III: Re-train Model

To re-train the model on Market1501 with original and occluded images, with GPU 0, run

```bash
bash script/experiment/train.sh OCCLUSION_TYPE market1501 0
```

where `OCCLUSION_TYPE` should be set to `No-Adversary`, `Random`, `Hard-1`, or `Sampling`.

# Citation

If you find our work useful, please kindly cite our paper:
```
@inproceedings{huang2018adversarially,
  title={Adversarially Occluded Samples for Person Re-Identification},
  author={Huang, Houjing and Li, Dangwei and Zhang, Zhang and Chen, Xiaotang and Huang, Kaiqi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5098--5107},
  year={2018}
}
```
