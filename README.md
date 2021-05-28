# Introduction

This repository contains the code for DropReg, which is a simple yet effective regularization method built upon dropout.
Through minimizing the bidirectional KL divergence of the outputs of any sub-model pairs sampled from dropout in model training, DropReg can constrain and regularize the output distributions of sub-models produced by the randomness of dropout.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version == 1.8
* Python version >= 3.6

**Installing from source**

To install fairseq from source and develop locally:
```
git clone https://github.com/dropreg/DropReg.git
cd DropReg
pip install --editable .
```

# Getting Started

**IWSLT'14 German to English (Transformer)**

First download and preprocess the data following example/translation.

Next we'll train a Transformer translation model with DropReg over this data:
```
cd example/drop_reg
bash run_train.sh
```
