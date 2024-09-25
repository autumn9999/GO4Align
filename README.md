# GO4Align

Welcome to the official repository for "GO4Align: Group Optimization for Multi-Task Alignment," one effective and efficient approach to multi-task optimization. 


###Project Webpage
Details will be available soon.

### Abstract

This paper proposes GO4Align, a multi-task optimization approach that tackles task imbalance by explicitly aligning the optimization across tasks. 
To achieve this, we design an adaptive group risk minimization strategy, compromising two crucial techniques in implementation:
- **dynamical group assignment**, which clusters similar tasks based on task interactions; 
- **risk-guided group indicators**, which exploit consistent task correlations with risk information from previous iterations. 

Comprehensive experimental results on diverse typical benchmarks demonstrate our method's performance superiority with even lower computational costs.

### Paper

[The preprint of our paper](https://arxiv.org/abs/2404.06486) is available on arXiv. 

### Framework of Adaptive Group Risk Minimization 

<p align="center"> 
    <img src="GO4Align.pdf" width="800">
</p>

---

###  Setup Environment 

We recommend using miniconda to create a virtual environment for running the code:
```bash
conda create -n go4align python=3.9.7
conda activate go4align 
conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=12.3 -c pytorch
conda install pyg -c pyg -c conda-forge
```

Install the package by running the following commands in the terminal:
```bash
git clone https://github.com/autumn9999/GO4Align.git
cd GO4Align
pip install -e .
```

GPU: NVIDIA A100-SXM4-40GB

###  Download Datasets

This work is evaluated on several multi-task learning benchmarks:

1. [NYUv2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) (3 tasks), where the link is provided by the previous MTO work [CAGrad](https://github.com/Cranial-XIX/CAGrad.git).
2. [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0) (2 tasks), where the link is provided by [CAGrad](https://github.com/Cranial-XIX/CAGrad.git).
3. [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) (40 tasks). Details can be found in the previous MTO work [FAMO](https://github.com/Cranial-XIX/FAMO).
4. QM9 (11 tasks), which can be downloaded automatically by `torch_geometric.datasets`. Details can be found in [FAMO](https://github.com/Cranial-XIX/FAMO/blob/main/experiments/quantum_chemistry/trainer.py).



###  Run Experiments
Here we provide experiments code for NYUv2. To run the experiment with other benchmark, please refer to unified APIs in [NashMTL](https://github.com/AvivNavon/nash-mtl) or [FAMO](https://github.com/Cranial-XIX/FAMO).

```bash
cd experiment/nyuv2
python trainer.py --method go4align 
```

We also support the following MTL methods as alternatives.

|    Method (code name)     |                                                          Paper (notes)                                                           |
|:-------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
|          Gradient-oriented methods            | -------------------------------------------------------------------------------------------------------------------------------- |
|           MGDA            |                     [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/pdf/1810.04650)                      |
|          PCGrad           |                           [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782)                           |
|          CAGrad           |                 [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf)                 |
|          IMTL-G           |                       [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr)                       |
|          NashMTL          |                        [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf)                        |
|          Loss-oriented methods            | -------------------------------------------------------------------------------------------------------------------------------- |
|            LS             |                                                       - (equal weighting)                                                        |
|            SI             |                                                - (see Nash-MTL paper for details)                                                |
|            RLW            |                  [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf)                  |
|            DWA            |                        [End-to-End Multi-Task Learning with Attention](https://arxiv.org/pdf/1803.10704)                         |
|            UW             | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
|           FAMO            |                          [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/pdf/2306.03792)                          |



### Citation
This repo is built upon [NashMTL](https://github.com/AvivNavon/nash-mtl) or [FAMO](https://github.com/Cranial-XIX/FAMO). If our work **GO4Align** is helpful in your research or projects, please cite the following papers:

```bib
@article{shen2024go4align,
  title={GO4Align: Group Optimization for Multi-Task Alignment},
  author={Shen, Jiayi and Wang, Cheems and Xiao, Zehao and Van Noord, Nanne and Worring, Marcel},
  journal={arXiv preprint arXiv:2404.06486},
  year={2024}
}

@article{liu2024famo,
  title={Famo: Fast adaptive multitask optimization},
  author={Liu, Bo and Feng, Yihao and Stone, Peter and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{navon2022multi,
  title={Multi-task learning as a bargaining game},
  author={Navon, Aviv and Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
  journal={arXiv preprint arXiv:2202.01017},
  year={2022}
}
```



