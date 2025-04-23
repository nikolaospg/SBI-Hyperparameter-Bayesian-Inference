from library import get_random_X, linear_model, quadratic_model, NN_model,\
generate_noise_vector, get_theoretical_posterior

import time 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import os 
import pymc as pm

#The function i use to do the model forward pass in the ABC loop.
#This also helps me choose the proper noise std for this specific model and prior (around 3.1)
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

    print("The variance of the non noisy output is ", np.var(Y), "the variance of the noise is ", noise_std**2)
    print("The SNR is ", np.var(Y)/noise_std**2)
    return output, current_seed




#Uses mcmc inference to get the posterior of the model specified
def mcmc_inference(X_obs, Y_obs, chain_length, target_accept, tune_percentage, theta_prior_mu, theta_prior_cov, model_type, noise_std=0.5):
    print("Running mcmc inference with chain length=%d target_accept=%f tune_percentage=%f model_type=%s noise_std=%f" % (chain_length, target_accept, tune_percentage, model_type, noise_std))
   
    def linear_model(X, theta):
        mu = theta[0] + theta[1] * X
        return mu
    
    def quadratic_model(X, theta):
        X1 = X[:,0]
        X2 = X[:,1]
        mu = (theta[0] + 
              theta[1] * X1 + 
              theta[2] * X2 + 
              theta[3] * X1**2 + 
              theta[4] * X2**2 + 
              theta[5] * X1 * X2)
        return mu
    
    def NN_model(X, theta):
        X1 = X[:, 0]
        X2 = X[:, 1]

        u1 = np.tanh(theta[0] * X1 + theta[1] * X2)
        u2 = np.tanh(theta[2] * X1 + theta[3] * X2)

        mu = theta[4] * u1 + theta[5] * u2
        return mu


    import logging
    t1 = time.time()

    with pm.Model() as model:
        # Define joint prior for 6 parameters using multivariate normal
        theta = pm.MvNormal("theta", mu=theta_prior_mu, cov=theta_prior_cov, shape=len(theta_prior_mu))

        # Now getting the mu, used later as the mean of the likelihood
        if(model_type == "linear"):
            mu = linear_model(X_obs, theta)
        elif(model_type == "quadratic"):
            mu = quadratic_model(X_obs, theta)
        elif(model_type == "NN"):
            mu = NN_model(X_obs, theta)
    
   
        # p(y | x, theta) = regresion_model(x, theta) + added_noise:
        likelihood = pm.Normal("y", mu=mu, sigma=noise_std, observed=Y_obs)

        # Sample from posterior using NUTS
        logging.getLogger("pymc").setLevel(logging.WARNING)
        trace = pm.sample(
            draws=chain_length,                               
            tune=int(chain_length * tune_percentage),                    
            discard_tuned_samples=True,
            chains=4,
            progressbar=True,
            target_accept=target_accept,
            cores = 4
        )

    t2 = time.time()
    print(pm.summary(trace))            #Summary describing the estimation
    MCMC_posterior = trace.posterior["theta"].values[0]  
    MCMC_means = np.mean(MCMC_posterior, axis=0)         
    MCMC_cov = np.cov(MCMC_posterior, rowvar=False, bias=True)     

    
    print("\nMCMC process time:", t2 - t1)
    print("\nResults from one chain, length=", len(MCMC_posterior))
    print("Posterior means:", MCMC_means)
    print("Posterior covariance:\n", MCMC_cov)


    MCMC_posterior_whole = trace.posterior["theta"].values.reshape(-1, trace.posterior["theta"].shape[-1])
    MCMC_means_whole = np.mean(MCMC_posterior_whole, axis=0)         
    MCMC_cov_whole = np.cov(MCMC_posterior_whole, rowvar=False, bias=True)   
    print("\nResults from all the chains length=", len(MCMC_posterior_whole))
    print("Posterior means:", MCMC_means_whole)
    print("Posterior covariance:\n", MCMC_cov_whole)
    return MCMC_posterior_whole, MCMC_means_whole, MCMC_cov_whole



def main():

    t1=time.time()

    #HYPERPARAMETERS - INITIALISATION OF THE ALGORITHM#
    chain_length = 10000
    noise_std=0.22                                                       #Known noise std (0.5 0.22)
    N=10000                                                                  #Number of observations 
    target_accept = 0.9
    tune_percentage = 0.25
    model_type = "NN"      
    if(model_type == "NN" or model_type == "quadratic"):                                           #"linear" "quadratic" or "NN"
        num_predictors = 2
        num_params = 6
        num_rows = 3            #Variable to create histograms later on
    elif(model_type == "linear"):
        num_predictors = 1
        num_params = 2
        num_rows = 1


    #Creating the prior, using sp.stats.

    # prior_means=np.array([-1, 2])
    # prior_covs=np.array([[1.01, 0.5], [0.5, 1.01]])   

    # prior_means = np.array([-1, 2, 0, 1, -2, 0.5])
    # prior_covs = np.array([
    #     [1.2, 0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.5, 1.5, 0.5, 0.4, 0.3, 0.2],
    #     [0.4, 0.5, 1.8, 0.5, 0.4, 0.3],
    #     [0.3, 0.4, 0.5, 1.1, 0.5, 0.4],
    #     [0.2, 0.3, 0.4, 0.5, 1.4, 0.5],
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 1.7]
    # ])


    prior_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prior_covs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    ])

    if(len(prior_means) != num_params):
        print("The model has %d params the prior you propose has %d" % (num_params, len(prior_means)))
        exit()
    prior_distr=sp.stats.multivariate_normal(mean=prior_means, cov= prior_covs)

    print("prior mean", prior_means)
    print("prior cov\n", prior_covs)
    #FINISHED WITH THE INITIALISATION#

    #GETTING GROUND TRUTH, EVIDENCE AND THE TARGET POSTERIOR (EITHER THEORETICAL OR MCMC INFERRED)#
    random_state_used=42                   #In case we want reproductibility - Set None if u do not want reproductibility
    

    ground_truth = prior_distr.rvs(size=1, random_state=random_state_used)         
    X_obs=get_random_X(N=N, num_predictors=num_predictors, random_state=random_state_used)
    Y_obs,_=simulate_output_ABC(X_obs, ground_truth, noise_std, model_type, random_state=random_state_used)

    print("the ground truth is ", ground_truth)
    print("X obs final element", X_obs[-1])
    print("Y obs final element", Y_obs[-1])

    if(model_type == "NN"):                 #If we have a NN we save/load the result
        path_to_file = "MCMC_posteriors/N=" + str(N) + ",noise=" + str(noise_std) + ".npy"
        if os.path.exists(path_to_file):
            print("Loading MCMC posterior from File with path ",path_to_file)
            MCMC_posterior = np.load(path_to_file)
            posterior_means = np.mean(MCMC_posterior, axis=0)         
            posterior_cov = np.cov(MCMC_posterior, rowvar=False, bias=True)  
            print("Posterior means:", posterior_means)
            print("Posterior covariance:\n", posterior_cov) 
            
        else: 
            print("File with path ",path_to_file, "Not found, doing the MCMC inference and saving on path", path_to_file)
            MCMC_posterior, MCMC_posterior_means, MCMC_posterior_covs=mcmc_inference(X_obs, Y_obs, chain_length, target_accept, tune_percentage, prior_means, prior_covs, model_type, noise_std) 
            np.save(path_to_file, MCMC_posterior)

    elif(model_type=="linear" or model_type=="quadratic"):              #Else we calculate result and compare with theoretical posterior
        theoretical_posterior_means, theoretical_posterior_covs=get_theoretical_posterior(prior_means, prior_covs, noise_std, X_obs, Y_obs, model_type)  
        MCMC_posterior, MCMC_posterior_means, MCMC_posterior_covs=mcmc_inference(X_obs, Y_obs, chain_length, target_accept, tune_percentage, prior_means, prior_covs, model_type, noise_std)
        print("\n\nTheoretical results:")
        print("mean", theoretical_posterior_means)
        print("cov\n", theoretical_posterior_covs)
        print("\nMCMC results")
        print("mean", MCMC_posterior_means)
        print("cov\n", MCMC_posterior_covs)   

    #Plotting the results:
    print(MCMC_posterior.shape)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_params):
        axes[i].hist(MCMC_posterior[:, i], bins=100, edgecolor='black')
        axes[i].set_title("Histogram of " + f'θ[{i}]')
        axes[i].set_ylabel('Frequency')

    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.show()




if __name__ == "__main__":
    main() 

