# EC-Depth
The official Pytorch implementation of the paper:
> **EC-Depth: Exploring the consistency of self-supervised monocular depth estimation under challenging scenes**
>


## Installation
Please refer to [dataset_prepare.md](./docs/dataset_prepare.md) for dataset preparation and [get_started.md](./docs/get_started.md) for installation.

## Running

We provide example bash commands to run training or testing. Please modify these files according to your own configuration before running.

### Training

First stage training:

```bash
bash train_first_stage.sh train first_stage_model 2 4 
```
Second stage training:

```bash
bash train_second_stage.sh train second_stage_model 2 4 
```

### Testing

Evaluate the model on KITTI dataset：

```bash
bash evaluate_kitti.sh
```

Evaluate the model on KITTI-C dataset：

```bash
bash evaluate_kittic.sh
```

## Results
We provide the official weights of EC-Depth (the first stage model) and EC-Depth* (the second stage model) on [Google Drive](https://drive.google.com/drive/folders/1UCmVxb640s4xukMyGZx5ib67SQgpmquI?usp=drive_link). Their experimental results on KITTI and KITTI-C are as below. 

### KITTI
| Methods  | AbsRel  | SqRel  | RMSE  | RMSE log  | a1  | a2  | a3  |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| EC-Depth  | 0.100 | 0.708 | 4.367 | 0.175 | 0.896 | 0.966 | 0.984 |  |
| EC-Depth*  | 0.100 | 0.689 | 4.315 | 0.173 | 0.896 | 0.967 | 0.985 |

### KITTI-C
| Methods  | AbsRel  | SqRel  | RMSE  | RMSE log  | a1  | a2  | a3  | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| EC-Depth  | 0.115 | 0.841 | 4.749 | 0.189 | 0.869 | 0.958 | 0.982 |  |
| EC-Depth*  | 0.111 | 0.807 | 4.651 | 0.185 | 0.874 | 0.960 | 0.983 |


## Acknowledgements

The code is based on [MonoDepth2](https://github.com/ldkong1205/RoboDepth), [MonoViT](https://github.com/zxcqlf/MonoViT), and [RoboDepth](https://github.com/ldkong1205/RoboDepth). 