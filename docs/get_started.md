# Get Started
## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch. We ran our experiments with PyTorch 1.7.1, CUDA 11.0, Python 3.7 and Ubuntu 18.04. We recommend using the same configuration to avoid environment conflicts.

**Note:**
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install [Anaconda](https://www.anaconda.com/download#downloads) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) from the official website.

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name ecdepth python=3.7 -y
conda activate ecdepth
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.


```shell
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
```


## Installation

We recommend that users follow our practices for installation.


**Step 1.** Install requirements.

```shell
git clone https://github.com/RuijieZhu94/EC-Depth.git
cd EC-Depth
pip install -r requirements.txt
```

**Step 2.** Install optional requirements (for training).

```shell
pip install -r requirements_optional.txt
```

Download checkpoints for [MPViT encoder pretrained on ImageNet-1K](https://github.com/youngwanLEE/MPViT#main-results-on-imagenet-1k), e.g.
```shell
mkdir ckpt
cd ckpt
wget https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth # mpvit-small
```

