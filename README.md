# MIL for SMI

***
## Introduction
This repository is for the proposed MIL used for Severe Mental Illness in Clincal Routine MRI.

Zhang W., Yang C., Cao Z., Li Z., Zhuo L., Tan Y., He Y., Yao L., Zhou Q., Gong Q., Sweeney J., Shi F., & Lui S. Detecting individuals with severe mental illness using artificial intelligence applied to magnetic resonance imaging.

***
## Content
* data: this folder contains five subjects along with whole procedure.
* train.py: contains the functions to train the network, and also needs to be fitness to your task and data with some changes.
* test.py: contain the functions to test the trained model, and also needs to be fitness to your task and data with some changes.
* config.py: contain configurations during the training.
* network.py: contains the structure of our model.
* AttLoss.py: implements partly difference loss functions used in our training processing

***
## Requirement
This repository is based on PyTorch 1.1.0, developed in Ubuntu 16.04 environment.
* cuda (required by pytorch), cudnn, numpy, scipy, sklearn, tqdm, pillow, matplotlib, ipython

***
Note that due to potential commercial issue, the trained model on large dataset was not included.

