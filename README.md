# Density-aware Chamfer Distance

### Todo
- [x] change the name of parameters (T -> alpha).
- [x] check the category name and number
- [x] loss print ("loss outlier")
- [x] maybe remove the test file and leave only train with --test_only
- [x] some args (e.g., flag) 
- [x] up-sampling
- [ ] modify the data path [here]()
- [x] test vrc_dcd pre-trained model
- [x] add pytorch 1.5 implementation
- [x] batch size
- [x] remove simple_eval
- [ ] setup (main setup and >1.5 instruction)

This repository contains the PyTorch implementation of the [paper](): 

*Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion, NeurIPS 2021*

[Tong Wu](https://github.com/wutong16), [Liang Pan](https://scholar.google.com/citations?user=lSDISOcAAAAJ), [Junzhe Zhang](https://junzhezhang.github.io/), [Tai Wang](https://tai-wang.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Dahua Lin](http://dahua.me/)

<img src='./assets/github_teaser.png' width=800>

> We present a new point cloud similarity measure named Density-aware Chamfer Distance (DCD). It is derived from CD and benefits from several desirable properties: **1)** it can detect disparity of density distributions and is thus a more intensive measure of similarity compared to CD; **2)** it is stricter with detailed structures and significantly more computationally efficient than EMD; **3)** the bounded value range encourages a more stable and reasonable evaluation over the whole test set. 
> DCD can be used as both an evaluation metric and the training loss. We mainly validate its performance on point cloud completion in our paper.

This repository includes:
- Implementation of Density-aware Chamfer Distance (DCD).
- Implementation of our method for this task and the pre-trained model.

## Environment 
* [PyTorch](https://pytorch.org/) (tested on 1.2.0)
* Other dependencies partially listed in `requirements.txt` 

## Datasets
We use the [MVP Dataset](https://mvp-dataset.github.io/). Please download the [train set](https://drive.google.com/file/d/1bY2RfPj_DvviNpr6ZzrEqhl4f7fMIqPF/view?usp=sharing) and [test set](https://drive.google.com/file/d/1qJT4uNURyDnPb_tI2vAntT2Iq98lhQMi/view?usp=sharing) and modify the data path [here]() to the your own data location.
Please refer to the [codebase](https://github.com/paul007pl/MVP_Benchmark) for further instructions.

## Usage
  + To train a model: run `python train.py ./cfgs/*.yaml`, e.g. `python train.py ./cfgs/vrc_plus.yaml`
  + To test a model: run `python train.py ./cfgs/*.yaml --test_only`, e.g. `python train.py ./cfgs/vrc_plus_eval.yaml --test_only`
  + Config for each algorithm can be found in `cfgs/`.
  + `run_train.sh` and `run_test.sh` are provided for SLURM users. 

We provide the following config files:
- `pcn.yaml`: the original PCN training with CD loss.
- `vrc.yaml`: the original VRCNet training with CD loss.
- `pcn_dcd.yaml`: using DCD as the loss function for PCN.
- `vrc_dcd.yaml`: using DCD as the loss function for VRCNet.
- `vrc_plus.yaml`: training our method.
- `vrc_plus_eval.yaml`: testing our method with guided down-sampling.


**Attention:**
We empirically find that using [DP](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) or [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) for training would slightly hurt the performance. So training on multiple cards is not well supported currently.


### Pre-trained model
We provide the [pre-trained model](https://drive.google.com/file/d/1WQFgxFQj3a-SkDaViCk3VqBE9Y_uZysG/view?usp=sharing) that reproduce the results in our paper.
Download and extract them to the `./log/pretrained/` directory, and then evaluate it with `cfgs/vrc_plus_eval.yaml`. The setting `prob_sample: True` turns on the guided down-sampling.


## Citation
If you find our code or paper useful, please cite our paper:
```bibtex
@inproceedings{dcd2021wu,
  title={Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion},
  author={Tong Wu, Liang Pan, Junzhe Zhang, Tai WANG, Ziwei Liu, Dahua Lin},
  booktitle={In Advances in Neural Information Processing Systems (NeurIPS), 2021},
  year={2021}
}
```
## Acknowledgement
The code is based on the [VRCNet](https://github.com/paul007pl/VRCNet) implementation. We include the following PyTorch 3rd-party libraries: 
[ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), 
[emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion), and 
[Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch).
Thanks for these great projects.

## Contact
Please contact [@wutong16](https://github.com/wutong16) for questions, comments and reporting bugs.


