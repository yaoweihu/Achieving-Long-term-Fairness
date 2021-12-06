# Achieving-Long-term-Fairness

This repository is the implementation of Achieving Long-term Fairness in Sequential Decision Making.

## Overview

+ `data` contains the Taiwan dataset.
+ `exp` contains the codes and results of experiments.
+ `src` contains the source codes.

## Requirements and dependencies
This project was designed with Python 3.8.5. We haven't tested the codes with other versions.

Required python pakcages:

+ numpy
+ pandas
+ cvxpy
+ pytorch
+ matplotlit
+ scikit-learn

## Experiment Environment

+ OS: Linux (Ubuntu-18.04)
+ CPU: Intel(R) Core(TM) i7-9700K 3.60GHz
+ Memory: 16GB
+ Iteration of retraining: 30 times for synthetic dataset and 50 times for semi-synthetic dataset

## Hyperparameter Settings

In our settings, lambda_u = 1, \lambda_l = 0.154 and \lambda_s = 0.119 for the synthetic data and lambda_u = 1, \lambda_l = 0.715 and \lambda_s = 0.022 for the synthetic data.
These hyperparameters can be normalized to get the sum 1. 
