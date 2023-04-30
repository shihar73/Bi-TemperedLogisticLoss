# bi-tempered-loss-pytorch

This code implements the bi_tempered_logistic_loss function from the paper "Bi-Tempered Logistic Loss for Training Neural Nets with Noisy Data" by Demis Hassabis, Neil Rabinowitz, Yee Whye Teh, and Aäron van den Oord. The function is used for training neural networks on classification tasks, particularly in the presence of noisy data.

This repository contains the direct translation from Tensorflow to Pytorch of the paper "Robust Bi-Tempered Logistic Loss
Based on Bregman Divergences" (https://arxiv.org/abs/1906.03361). The repo translates the official Tensorflow repo which can be found here: https://github.com/google/bi-tempered-loss

## Code Description
The code consists of two Python files:

- bi_tempered_loss.py: This file contains the implementation of the bi_tempered_logistic_loss function.
- task1.py: This file uses the bi_tempered_logistic_loss function to train a neural network on a classification dataset and visualize the results.

## Key Takeaways from "Bi-Tempered Logistic Loss for Training Neural Nets with Noisy Data"

- The bi_tempered_logistic_loss function is a loss function that is specifically designed to handle noisy data in classification tasks.
- The function introduces two temperature parameters that control the sharpness of the softmax function used in the loss calculation.
- The function has been shown to outperform other loss functions on datasets with noisy labels.