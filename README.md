# SBI-Hyperparameter-Bayesian-Inference
Using Simulation Based Inference techniques (namely SMC ABC) to estimate the posterior distribution of hyperparameters of Machine Learning models. 

regression_analysis -> Applies the SMC ABC method to do Bayesian Inference for Regression Problems of one output (Linear, Quadratic, NN). For the Linear and Quadratic models, the theoretical gaussian posterior moments are known in closed form equations. For the NN we do MCMC inference (NUTS) with the help of PyMC to have a baseline for comparison.
