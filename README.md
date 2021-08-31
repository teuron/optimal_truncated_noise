# Learning Numeric Optimal DifferentiallyPrivate Truncated Additive Mechanisms

Implementation for the publication "Learning Numeric Optimal Differentially Private Truncated Additive Mechanisms" by Sommer, Abfalterer, Zingg, and Mohammadi, Where, 2021 [URL](URL) [1]

### Introduction

Differentially private (DP) mechanisms face the challenge of providing accurate results while protecting their inputs: the privacy-utility trade-off. A simple but powerful technique for DP adds noise to sensitivity-bounded query outputs to blur the exact query output: additive mechanisms. While a vast body of work considers infinitely wide noise distributions, some applications (e.g., real-time operating systems) require hard bounds on the deviations from the real query, and only limited work on such mechanisms exist. An additive mechanism with truncated noise (i.e., with bounded range) can offer such hard bounds. 

This repository contains the implmentation of our gradient-descent-based tool to learn truncated noise for additive mechanisms with strong utility bounds while simultaneously optimizing for differential privacy under sequential compositions (e.g. [Experiment on optimality for sequential compositions](./experiments/optimality_sequential_compositions/generator.py)). As illustrated in our work [1], this tool finds noise patterns that are remarkably close to truncated Gaussians and even replicate their shape for L2 utility-loss. Learning optimal noise for DP-SGD (sub-sampling) shows similar effects (e.g. [Experiment on DP-SGD](./experiments/dp_sgd/generator.py)). 

### Directory / 

Contains the implementation of our model using PyTorch. The Sigmoid model that we use for noise generating is located in [/models.py](./models.py#L159).

#### Directory /experiments

The directory contains all experiments we conducted. To rerun one experiment, please use `python3 generator.py` in the corresponding directory. 

#### File evaluator.py

Contains the logic for running a specific experiment using the parameter from the below table. An example evaluation can be started with

```
    python3 evaluator.py --noise_model SigmoidModel --method renyi_markov --number_of_compositions 1 --utility_weight 0.5 --range_begin 500 --random_init --sig_num 10000 --epochs 100000 --element_size 60000

```

| Parameter                     | Usage         |  Description                                                                  |
| ----------------------------- | ------------- | ----------------------------------------------------------------------------- |
| pt_seed	                    |  Global       |  Sets the seed.                                                               |
| eps	                        |  Global       |  Sets eps. Default: 0.3                                                       |
| range_begin	                |  Global       |  Sets r. Default: 15                                                          |
| element_size	                |  Global       |  Sets the amount of elements. Default: 60000                                  |
| number_of_compositions	    |  Global       |  Sets n. Default: 1                                                           |
| epochs	                    |  Global       |  Sets the training epochs. Default: 15000                                     |
| noise_class	                |  Global       |  Either "SymmetricNoise" or DP-SGD "MixtureNoise"                             |
| mixture_q	                    |  Global       |  Sets q for DP-SGD "MixtureNoise"                                             |
| bias	                        |  model        |  Sets the bias of the Sigmoid model. Default: 10                              |
| slope	                        |  model        |  Sets the slope of the Sigmoid model. Default: 500                            |
| sig_num	                    |  model        |  Sets the amount of Sigmoid functions used. Default: 10.000	                |
| method	                    |  criterion    |  Sets the Privacy Accountant method.                                          |
| utility_loss_function	        |  criterion    |  Either L_1 or L_2 utility loss. Default: L_1                                 |
| utility_weight	            |  criterion    |  Sets the utility weight. Default: 0.1                                        |
| utility_weight_decay	        |  criterion    |  Use utility weight decay. Default: False                                     |
| utility_weight_halving_epochs	|  criterion    |  Sets the amount of epochs needed to half the utility weight                  |
| utility_weight_minimum	    |  criterion    |  Minimum utility weight after halving. Default: 0.0001                        |
| learning_rate	                |  optimizer    |  Controls the learning rate. Default: 0.001                                   |
| learning_rate_decay	        |  lr_decay     |  Controls the exponential learning rate decay. Usually between \[0.99995, 1\] |
| pb_buckets_half               |  PB           |  Number of Privacy Buckets. Default: 500                                      |
| pb_factor                     |  PB           |  Privacy Buckets coarseness. Default: 1.00001                                 |
	



### Directory /gsl
Contains version 2.6 of the GNU Scientific Library. Run `make` in the directory to build GSL and `make clean` if you want to remove it.

## References
