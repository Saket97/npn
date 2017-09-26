# npn
Neural Networks require a massive amountof data for training, so stochastic algorithms are generally used for training.  But, neural networksare prone to overfitting, especially when training data is insufficient.  Also there is no estimate ofour  confidence  in  our  prediction.   Bayesian  methods  can  be  helpful  in  tackling  these  problems.Wang et al(2016) proposed one such application named Natural Parameters Networks in NIPS2016.Natural parameter networks make use of the properties of the exponential families to efficientlytrain the neural networks using backpropagation, while maintaining the flexibility of the network i.e.allowing the modelling of data using distributions other than Gaussian.

This repository contains the implementation of the Natrual Parameter Networks paper. We have implemented 2 versions: Gamma NPN, and Gaussian NPN.

We have also **extended** the paper for **Recurrent Natural Parameter Networks**, and implemented the Gaussian RNPN.
