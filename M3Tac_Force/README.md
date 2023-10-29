# FSwin_MAP
This repo provides the PyTorch implementation of the work FSwin_MAP.

## Overview
<p align="center">
<img src='assets/Network.png' width='800'>
<p>

## Requirements

* Pytorch 1.8.1,torchvision
* numpy 1.23.5
* opencv-python 4.7.0.72
* pandas 2.0.0
* tqdm 4.65.0 
* tensorboardx 2.6
## Datasets
Please download the dataset from the following link:

https://cloud.tsinghua.edu.cn/d/a6d80af5c0664250be0d/

The structure of the dataset is as follows:

.
└── Dataset
    ├── new_map
    │         ├── 0.5.txt
    │         ├── 1.txt
    │         ├── 1.5.txt
    │         ├── 2.txt
    │         ├── 2.5.txt
    │         └── 3.txt
    ├── test0.5
    │         ├── inf_cut
    │         ├── mask
    │         └── data_force.txt
    ├── test1
    ├── test1.5
    ├── test2
    ├── test2.5
    ├── test3


## Training

Before you start training, make sure you have downloaded and processed the dataset correctly.
To make it easier to adjust the training parameters, we have stored all the parameters in the main.py file. If you want to start training as soon as possible, please set data_path to the path of the dataset you downloaded and stored.

`data_path = "path_to_your_dataset"`

Example:

`data_path = "dataset/your_data"`

Other parameters can be modified in the same place in the main.py file.

## Evaluation

Evaluation of the code can be done in the inference.py file, which has the same parameter structure as main.py.

Change the checkout_path parameter to the path where you store the weight files to be evaluated:

`checkout_path = "path_to_your_checkpoint"`

As in training, change the data_path argument to the path where you store the dataset:

`data_path = "path_to_your_dataset"`

When the evaluation is completed, the results will be printed directly.