# BC-GNN
Zero-Shot Learning of Aerosol Optical Properties with Graph Neural Networks

## Overview
This repository contains a PyTorch implementation of the code for the paper "Zero-Shot learning of aerosol optical properties with graph neural networks".

## Citation

The preprint for this paper can be found at:

```
@article{Lamb2021,
  title={Unsupervised Learning of Predictors for Microphysical Process Rates},
  author={Lamb, K.D. and P. Gentine},
  journal={arXiv preprint arXiv:2107.10197},
  year={2021}
}
```

## Content
- [Training Data Generation](#training-data-generation)
- [Data Preparation](#data-preparation)
- [Training the Model](#training)
- [Evaluating the Network](#evaluation)

## Training Data Generation
Cartesian coordinates for multi-sphere clusters are generated with a cluster-cluster algorithm. 

Training data is generated by running the Fortran-90 implementation of the Multiple Sphere T-Matrix code (Mackowski and Mischenko, 2011). Example shell scripts for developing the training data code are found in the MSTMscripts folder.

## Data Preparation

## Training the Model

## Evaluating the Network



