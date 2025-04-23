import pytensor
pytensor.config.mode="NUMBA" 
pytensor.config.cxx=""              #To use blas

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import binom
from tabulate import tabulate

import scipy as sp
import sys
import time

import pandas as pd
import statsmodels.api as sm
import seaborn as sns

from numba import njit
import math
import os 


#Checks if the matrix is symmetric and positive definite:
def is_legit_cov_matrix(matrix):
    
    # Check if the matrix is symmetric:
    if not np.allclose(matrix, matrix.T):
        print("Covariance matrix is not symmetric")
        return False  # Not symmetric → Not a valid covariance matrix
    
    # Check if the matrix is positive semi-definite (PSD)
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(eigenvalues < 0):
        print("Covariance matrix is not positive semi definite")
        return False  # Negative eigenvalues → Not a valid covariance matrix

    return True  # Symmetric and PSD → Valid covariance matrix

#Abstraction to easily get a gaussian prior. 
def create_gaussian_prior(model_type):

    if(model_type == "linear"):                          
        prior_means=np.array([-1, 2])                   
        prior_covs=np.array([[1.01, 0.5], [0.5, 1.01]])    
        noise_std = 0.5                             
    
    elif(model_type == "NN"):
        prior_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        prior_covs = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
        ])
        noise_std = 0.22         
          
    elif(model_type == "quadratic"):
        prior_means = np.array([-1, 2, 0, 1, -2, 0.5])
        prior_covs = np.array([
            [1.2, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.5, 1.5, 0.5, 0.4, 0.3, 0.2],
            [0.4, 0.5, 1.8, 0.5, 0.4, 0.3],
            [0.3, 0.4, 0.5, 1.1, 0.5, 0.4],
            [0.2, 0.3, 0.4, 0.5, 1.4, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 1.7]
        ])
        # noise_std = 0.75
        noise_std = 0.5


    else:
        print("Do not have prior values for this specified model, exiting")
        exit()

    #Before returning the values we must check that the prior_covs is a legitimate covariance matrix
    if(is_legit_cov_matrix):
        return prior_means,prior_covs,noise_std
    else:
        exit()
    
#Function we use to get the X_obs... uses uniform distribution
def get_random_X(N, num_predictors, left_val=0, right_val=2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)  # Set the seed using the older RandomState
    if(num_predictors>1):
        X = np.random.uniform(low=left_val, high=right_val, size=(N,num_predictors))
    else:
        X = np.random.uniform(low=left_val, high=right_val, size=(N))
    return X

#Function that generates a vector of gaussian noise. Also returns the seed used.
def generate_noise_vector(noise_std, size, random_state = None):
    if(random_state is None):
        my_seed = np.random.randint(0, 2**32 - 1)
    else:
        my_seed = random_state
    np.random.seed(my_seed)                                 #Set the seed
    noise = np.random.normal(0, noise_std, size)          #Noise vector
    return noise, my_seed

#JIT function that implements the inference of X dataset on linear model
@njit
def linear_model(X, theta_vector):

    Y = theta_vector[0] + theta_vector[1] * X 
    
    return Y

#JIT function that implements the inference of X dataset on quadratic model
@njit
def quadratic_model(X, theta_vector):
    X1 = X[:, 0]  
    X2 = X[:, 1]  

    Y = (theta_vector[0] + 
         theta_vector[1] * X1 + 
         theta_vector[2] * X2 + 
         theta_vector[3] * X1 * X1 + 
         theta_vector[4] * X2 * X2 + 
         theta_vector[5] * X1 * X2)
    return Y
    
#JIT function that implement the inference of X dataset from a NN, with 1 hidden layer containing 2 tanh neurons and linear neuron on the output
#In total 6 parameters, no intercepts, X has two features (columns)
@njit(fastmath=True)
def NN_model(X, theta_vector):
    X1 = X[:, 0]
    X2 = X[:, 1]

    u1 = np.tanh(theta_vector[0] * X1 + theta_vector[1] * X2)
    u2 = np.tanh(theta_vector[2] * X1 + theta_vector[3] * X2)

    Y = theta_vector[4] * u1 + theta_vector[5] * u2
    return Y


#Version of the function to be called inside njitted simulate_particles, where the X will be passed as a 2D array (numba wants consistency in dimension of arguments)
@njit
def linear_model_numba(X, theta_vector):

    x1 = X[:, 0] if X.ndim == 2 else X
    return theta_vector[0] + theta_vector[1] * x1


#Applying the Bayesian Linear Regression Theoretical Formula:
def get_theoretical_posterior(prior_means, prior_covs, noise_std, X_obs, Y_obs, model_type):          

    N = len(X_obs)
    
    #Getting the phi matrix 
    if(model_type == "linear"):
        phi=np.column_stack( [np.ones(shape=(len(X_obs),), dtype=X_obs.dtype), X_obs])
    if(model_type == "quadratic"):
            X1 = X_obs[:, 0]
            X2 = X_obs[:, 1]
            phi = np.column_stack([
            np.ones(N),       # theta_0
            X1,               # theta_1 * X1
            X2,               # theta_2 * X2
            X1**2,            # theta_3 * X1^2
            X2**2,            # theta_4 * X2^2
            X1 * X2           # theta_5 * X1*X2
        ])

    #Getting complementary variables to implement the equations given by bishop
    beta=1/(noise_std*noise_std)            #beta is the inverse of the noise variance, Bishop's book
    t=Y_obs
    m0=prior_means
    S0_inv = np.linalg.inv(prior_covs)
    phi_trans=np.transpose(phi)
    #Finished with the complementary variables of the bishop notation

    #Now calculating the posterior with the help of the evidence X_obs, Y_obs
    SN_inv = S0_inv + beta * phi_trans @ phi
    SN=np.linalg.inv(SN_inv)                #The covariance matrix of the posterior
    mN=SN @ (S0_inv @ m0 + beta*phi_trans @ Y_obs)
    print("\nThe theoretical posterior has \nmeans:",mN)
    print("and covariance :\n",SN)

    return mN, SN


#Calculating KL divergence between D1||D2 that are considered gaussian
def kl_divergence_gaussians(mu1, sigma1, mu2, sigma2):

    n = mu1.shape[0]
    sigma2_inv = np.linalg.inv(sigma2)
    diff = mu2 - mu1

    term1 = np.trace(sigma2_inv @ sigma1)
    term2 = diff.T @ sigma2_inv @ diff
    term3 = -np.log(np.linalg.det(sigma1) / np.linalg.det(sigma2))
    
    return 0.5 * (term1 + term2 + term3 - n)


@njit
def get_summary_statistics(Y, summary_method_param, JSD_bin_edges):
    
    #algorithm to get percentile on a sorted array.
    #It first finds the "float" index corresponding to the percentile. It then gets the floor,ceiling integer indices and the corresponding values
    #It finally get a linear interpolation to get the wanted value. More nuanced than just getting the two integers and getting the mean.
    def fast_percentile(sorted_array, percent):

        
        n = len(sorted_array)

        float_index = (n - 1) * (percent / 100.0)
        floor = int(float_index)  
        frac = float_index - floor  
        
        # If the index is at the boundary
        if floor == n - 1:
            return sorted_array[n - 1]
        elif floor ==0:
            return sorted_array[0]
        
        # Linear interpolation between adjacent values
        lower_value = sorted_array[floor]
        upper_value = sorted_array[floor + 1]
        return lower_value + frac * (upper_value - lower_value)

    if(summary_method_param == "raw"):
        return Y

    if(summary_method_param == "moments"):
        mean = np.mean(Y)
        min_val = np.min(Y)
        max_val = np.max(Y)
        n = len(Y)

        #First getting std, from bessel corrected var estimate:
        unbiased_sample_variance = np.sum((Y - mean) ** 2) / (n - 1)
        std = np.sqrt(unbiased_sample_variance)                  # Sample standard deviation
        
        #Adjusted skewness:
        skewness = np.sum(((Y - mean) / std) ** 3) / n
        adjusted_skewness = (np.sqrt(n * (n - 1)) / (n - 2)) * skewness

        #Adjusted kurtosis:
        kurtosis = np.sum(((Y - mean) / std) ** 4) / n
        excess_kurtosis = kurtosis - 3  # Excess kurtosis (relative to normal distribution)
        adjusted_kurtosis = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * excess_kurtosis + 6)
        moments_array = np.array([mean, std, adjusted_skewness, adjusted_kurtosis, min_val, max_val])
        return moments_array
    
    if(summary_method_param == "percentiles"):           
        mean = np.mean(Y)
        n = len(Y)

        #First getting std, from bessel corrected var estimate:
        unbiased_sample_variance = np.sum((Y - mean) ** 2) / (n - 1)
        std = np.sqrt(unbiased_sample_variance)                  # Sample standard deviation
        
        #Adjusted skewness:
        skewness = np.sum(((Y - mean) / std) ** 3) / n
        adjusted_skewness = (np.sqrt(n * (n - 1)) / (n - 2)) * skewness

        #Adjusted kurtosis:
        kurtosis = np.sum(((Y - mean) / std) ** 4) / n
        excess_kurtosis = kurtosis - 3  # Excess kurtosis (relative to normal distribution)
        adjusted_kurtosis = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * excess_kurtosis + 6)
        moments_array = np.array([mean, std, adjusted_skewness, adjusted_kurtosis])

        #Now getting percentiles
        # num_percentiles = int(np.sqrt(n))           #getting amount of percentiles by sqrt(N) +1 rule. Change this if you want to
        num_percentiles = 10 
        percentile_array = np.zeros(shape=(num_percentiles +1,))                
        sorted_Y = np.sort(Y)
        alpha=100/num_percentiles
        for current_index in range(len(percentile_array)):
            current_percent = current_index * alpha
            current_percentile = fast_percentile(sorted_Y, current_percent)
            percentile_array[current_index] = current_percentile
        # return percentile_array
        return np.append(moments_array, percentile_array)

    if(summary_method_param == "JSD"):
        
        # # CREATION OF HISTOGRAM, DISCARDING VAlUES OUTSIDE THE BIN EDGES#
        # n_bins = len(JSD_bin_edges) - 1
        # hist   = np.zeros(n_bins, dtype=np.float64)
        # count  = 0

        # for current_Y_val in Y:                               # numba-friendly loop
        #     # index of the *right* insertion point, then shift to the left edge
        #     current_index = np.searchsorted(JSD_bin_edges, current_Y_val, side='right') - 1
        #     # keep only in‑range hits
        #     if 0 <= current_index < n_bins:
        #         hist[current_index] += 1.0
        #         count    += 1

        # # convert to relative frequencies                      
        # inverse_count = 1.0 / count
        # for k in range(n_bins):                 
        #     hist[k] *= inverse_count

        # return hist

        #CREATION OF HISTOGRAM WITH 2 EXTRA BINS FOR UNDERFLOW AND OVERFLOW#
        total_num_bins  = len(JSD_bin_edges) + 1                  
        hist     = np.zeros(total_num_bins, dtype=np.float64)
        count    = 0

        left_edge  = JSD_bin_edges[0]
        right_edge = JSD_bin_edges[-1]

        for current_Y in Y:
            if current_Y < left_edge:
                hist[0] += 1.0                      
            elif current_Y >= right_edge:                    
                hist[-1] += 1.0                     
            else:
                current_index = np.searchsorted(JSD_bin_edges, current_Y, side='right') - 1
                hist[current_index + 1] += 1.0                 
            count += 1

        inverse_count = 1.0 / count
        for k in range(total_num_bins):                 
            hist[k] *= inverse_count

        return hist




@njit
def distance_metric(observed_output_vector, simulated_output_vector, distance_type="rmse"):
    if(distance_type == "rmse"):
        mse = np.mean((observed_output_vector - simulated_output_vector) ** 2)
        return np.sqrt(mse)
    
    if(distance_type == "sqrt_JSD"):
        P = observed_output_vector
        Q = simulated_output_vector
        n = len(P)
        if len(Q) != n:
            raise ValueError("Arrays must have same length")
        
        M = (P + Q)/2
        
        # Compute KL divergences: D_KL(P || M) and D_KL(Q || M)
        kl_div_pm = 0.0
        kl_div_qm = 0.0
        for i in range(n):
            if P[i] > 0:  # Avoid log(0) or 0/0
                kl_div_pm += P[i] * np.log(P[i] / M[i])
            if Q[i] > 0:
                kl_div_qm += Q[i] * np.log(Q[i] / M[i])
        
        # JSD = (1/2) * (D_KL(P || M) + D_KL(Q || M))
        jsd = 0.5 * (kl_div_pm + kl_div_qm)
        return np.sqrt(jsd)


#I get the bin edges from the Y_obs. I either use the square root rule of freedman diaconis rule
#The sqrt rule is simple, just take num_bins=sqrt(n), where n is the number of samples. Then assign equal widths per bin.
#The Freedman diaconis rule finds the bin width through the formula 2*IQR/(n^(1/3)). 
def get_bin_edges(Y_obs, rule="FD"):            #rule="FD" or "SR"

    #Getting Num bins with Square Root rule 
    if(rule=="SR"):
        n=len(Y_obs)
        num_bins = int(np.round(np.sqrt(n)))
        total_data_range = np.max(Y_obs) - np.min(Y_obs)
        bin_width = total_data_range/num_bins
        bin_edges = np.zeros(shape=(num_bins+1,))
        count = np.min(Y_obs)
        for current_index, _ in enumerate(bin_edges):
            bin_edges[current_index] = count
            count += bin_width
        bin_edges[-1]=np.max(Y_obs)
        return bin_edges


    #Getting Bin Width with Freedman Diaconis rule
    elif(rule=="FD"):
        n=len(Y_obs)
        IQR = np.percentile(Y_obs, 75)- np.percentile(Y_obs, 25)
        total_data_range = np.max(Y_obs) - np.min(Y_obs)
        bin_width = 2*IQR/np.power(n,1/3)
        num_bins = int(np.round(total_data_range/bin_width))
        bin_edges = np.zeros(shape=(num_bins+1,))
        count = np.min(Y_obs)
        for current_index, _ in enumerate(bin_edges):
            bin_edges[current_index] = count
            count += bin_width
        bin_edges[-1]=np.max(Y_obs)
        return bin_edges
    


#The function generates noise, passes the X dataset through the model, and returns the sum.
#Also returns the seed used to generate the noice. 
#This way we can recreate the results if we want to (e.g. calculate the Yi values for post processing). If we use raw output storing the Y values is too memory demanding for large amount of simulations.
def simulate_output_ABC(X, theta_vector, noise_std, model_type, random_state=None):
    noise, current_seed = generate_noise_vector(noise_std, len(X), random_state)
    if(model_type=="linear"):
        Y = linear_model(X, theta_vector)
    if(model_type=="quadratic"):
        Y = quadratic_model(X, theta_vector)
    if(model_type=="NN"):
        Y = NN_model(X, theta_vector)
    output = Y + noise
    if(random_state==42):
        print("The variance of the non noisy output is ", np.var(Y), "the variance of the noise is ", noise_std**2)
        print("The SNR is ", np.var(Y)/noise_std**2)
    return output, current_seed

