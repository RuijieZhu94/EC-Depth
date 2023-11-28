# EC-Depth
The official Pytorch implementation the paper:
> **EC-Depth: Exploring the consistency of self-supervised monocular depth estimation under challenging scenes**
>
> *Ruijie Zhu, Ziyang Song, Chuxin Wang, Jianfeng He, Tianzhu Zhang*

[![arxiv](https://img.shields.io/badge/arXiv-2310.08044-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2310.08044)
[![project](https://img.shields.io/badge/website-project-yellowgreen.svg?style=plastic)](https://ruijiezhu94.github.io/ECDepth_page/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ec-depth-exploring-the-consistency-of-self/unsupervised-monocular-depth-estimation-on-7)](https://paperswithcode.com/sota/unsupervised-monocular-depth-estimation-on-7?p=ec-depth-exploring-the-consistency-of-self)



<p align="center">
<img src="assets/overview.jpg" width="97%"/>
</p>

> The two-stage training framework of EC-Depth. In the first stage, we train DepthNet and PoseNet with the perturbation-invariant depth consistency loss. In the second stage, we leverage the teacher network to generate pseudo labels and construct a distillation loss to train the student network. Notably, we propose a depth consistency-based filter (DC-Filter) and a geometric consistency-based filter (GC-Filter) to filter out unreliable pseudo labels.

# News
- **28 Nov. 2023**: The [project website](https://ruijiezhu94.github.io/ECDepth_page/) was released.
- **12 Oct. 2023**: [EC-Depth](https://arxiv.org/abs/2310.08044) released on arXiv. 



# Bibtex

If you like our work and use the codebase or models for your research, please cite our work as follows.

```
@article{zhu2023ecdepth,
  title={EC-Depth: Exploring the consistency of self-supervised monocular depth estimation under challenging scenes},
  author={Zhu, Ruijie and Song, Ziyang and Wang, Chuxin and He, Jianfeng and Zhang, Tianzhu},
  journal={arXiv preprint arXiv:2310.08044},
  year={2023}
}
```