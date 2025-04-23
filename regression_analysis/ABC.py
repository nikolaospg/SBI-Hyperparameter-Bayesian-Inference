from library import create_gaussian_prior, get_random_X, linear_model, quadratic_model, NN_model,\
generate_noise_vector, get_theoretical_posterior, kl_divergence_gaussians, get_summary_statistics,\
get_bin_edges, distance_metric, simulate_output_ABC
from library_present import plot_gaussian_distributions, plot_non_gaussian_distributions
from numba import njit
import time 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import os 



#Computes the distances for every sample of the prior_samples parameter.
def compute_distances(prior_samples, X_obs, Y_obs, noise_std, summary_method_param, model_type, distance_type, JSD_bin_edges):
    distance_metrics=np.zeros(shape=(len(prior_samples)))
    summary_Y_obs = get_summary_statistics(Y_obs, summary_method_param, JSD_bin_edges)
    seeds=np.zeros(shape=(len(prior_samples)), dtype=np.int64)                  #array to keep track of the seeds
    for index,current_sample in enumerate(prior_samples):
        current_Y,current_seed=simulate_output_ABC(X_obs, current_sample, noise_std, model_type)
        summary_current_Y = get_summary_statistics(current_Y, summary_method_param, JSD_bin_edges)
        current_metric=distance_metric(summary_Y_obs, summary_current_Y, distance_type)
        distance_metrics[index]=current_metric
        seeds[index]=current_seed
    return distance_metrics, seeds


#Applying the ABC algorithm. It returns the empirical posterior samples and the corresponding distances.
#The reason why we also keep track and return the seeds used for random noise generation is to be able to recreate the corresponding Y and summary_Y for the posterior samples.
#This is used for the post correction algorithm. If we decided to return the summary statistics themselves, then if the summary method was "raw" then the space complexity would be too much.
def ABC_loop(prior_distr, num_prior_samples, X_obs, Y_obs, noise_std, summary_method_param, model_type, distance_type, limit, JSD_bin_edges):

    prior_samples=prior_distr.rvs(size=num_prior_samples)                                                                   #Sampling prior samples
    distances,seeds=compute_distances(prior_samples, X_obs, Y_obs, noise_std, summary_method_param, model_type, distance_type, JSD_bin_edges)             #computing distances for the prior sample
    distances=np.array(distances)

    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_seeds = seeds[sorted_indices]
    sorted_prior_sample = prior_samples[sorted_indices]
        
    final_distances=sorted_distances[0:limit]
    final_seeds = sorted_seeds[0:limit]
    empirical_samples=sorted_prior_sample[0:limit]
    empirical_mean = np.mean(empirical_samples, axis=0)
    empirical_cov = np.cov(empirical_samples, rowvar=False, bias=True)

    return empirical_samples, empirical_mean, empirical_cov, final_distances, final_seeds


#Implementing Beaumont's 2002 Local Linear regression correction
def post_processing_ABC(posterior_samples, distances, seeds, summary_method_param, model_type, distance_type, Y_obs, X_obs, noise_std, JSD_bin_edges):

    t1=time.time()

    #Calculates the R2 coefficients for every parameter... this potentially shows how good the estimate is  
    def calculate_R2(posterior_samples, X, B_hat):

        theta_predicted = X @ B_hat             #The predicted theta, as they come from the local linear model
        
        # Number of parameter dimensions
        num_params = len(posterior_samples[0])
        
        # Calculate R^2 for each column
        R2_values = np.zeros(num_params)
        for parameter_index in range(num_params):
            observed_param_values = posterior_samples[:, parameter_index]  
            predicted_param_values = theta_predicted[:, parameter_index]  
            
            # Total sum of squares (variance around mean)
            mean_observed_param_values = np.mean(observed_param_values)
            SS_tot = np.sum((observed_param_values - mean_observed_param_values)**2)
            
            # Residual sum of squares (unexplained variance)
            SS_res = np.sum((observed_param_values - predicted_param_values)**2)
            
            R2_values[parameter_index] = 1 - (SS_res / SS_tot)
        print("R2 vals", R2_values)
        return R2_values

    #Function that uses the saved seeds, recreates the Y outputs and gets the summaries.
    def get_summaries(posterior_samples, seeds, X_obs, Y_obs, summary_method_param, model_type, noise_std, JSD_bin_edges):
        summary_Y_obs = get_summary_statistics(Y_obs, summary_method_param, JSD_bin_edges)
        summaries = np.zeros(shape=(len(posterior_samples), len(summary_Y_obs)))
        for current_index,current_sample in enumerate(posterior_samples):
            current_seed = seeds[current_index]
            current_Y,_ = simulate_output_ABC(X_obs, current_sample, noise_std, model_type, current_seed)
            current_summary = get_summary_statistics(current_Y, summary_method_param, JSD_bin_edges)
            summaries[current_index] = current_summary
        return summary_Y_obs, summaries

    #Creates the weight matrix in sparse form, then solves the local linear regression
    def solve_regression(posterior_samples, distances, summaries, summary_Y_obs):
        from scipy import sparse
        
        def weight_matrix_sparse(distances):            #Using epanechnikov kernel
            delta = np.max(distances)  # maximum distance
            # Calculate diagonal values
            t = distances
            kernel_vals = (1 - (t/delta)**2) / delta
            normalising_count = np.sum(kernel_vals)
            weights = kernel_vals / normalising_count
            
            # Create sparse diagonal matrix
            W = sparse.diags(weights, format='csr')  # Using CSR format for efficient matrix operations
            return W
        
        # Create sparse weight matrix
        W = weight_matrix_sparse(distances)
        
        # Prepare X matrix
        X = np.hstack([np.ones((len(summaries), 1)), summaries - summary_Y_obs])
        X_trans = X.T
        # Perform matrix operations
        XTW = X_trans @ W  
        XTWX = XTW @ X    
        XTW_theta = XTW @ posterior_samples
        
        # print(XTWX)
        # Solve the linear system
        B_hat = np.linalg.inv(XTWX) @ XTW_theta
        
        Q = B_hat[1:]      #I drop the first column of the solution. This is used to implement the correction of type posterior_samples - (summaries - summary_Y_obs) @ Q.
        return Q, X, B_hat

    print("\nPost processing Information:")
    print("max distances:", np.max(distances), "min distances:", np.min(distances), "std of distances: ", np.std(distances))
    summary_Y_obs, summaries = get_summaries(posterior_samples, seeds, X_obs, Y_obs, summary_method_param, model_type, noise_std, JSD_bin_edges)

    Q, X, B_hat=solve_regression(posterior_samples, distances, summaries, summary_Y_obs)    
    corrected_samples = posterior_samples - (summaries - summary_Y_obs) @ Q             #This applies the correction
    calculate_R2(posterior_samples, X, B_hat)
    empirical_mean = np.mean(corrected_samples, axis=0)
    empirical_cov = np.cov(corrected_samples, rowvar=False)
    t2=time.time()
    print("Time for correction", t2-t1)
    print("corrected Means", empirical_mean)
    print("corrected covs", empirical_cov)

    return corrected_samples, empirical_mean, empirical_cov





def main():

    t1=time.time()

    #HYPERPARAMETERS - INITIALISATION OF THE ALGORITHM#
    num_prior_samples=100000
    percentile_threshold=1                                            #Percentile of best elements we keep
    num_posterior_samples=int(0.01*percentile_threshold*num_prior_samples)                      #The size of the empirical posterior distribution
    # MCMC_posterior_size=int(1*num_posterior_samples)
    MCMC_posterior_size = 10000
    N=1000                                                                  #Number of observations 
    post_process_flag = False                                           #whether to apply post processing regression correction
   
    model_type="quadratic"                                                 #"linear" "quadratic" or "NN"
    if(model_type == "linear"):
        num_params=2
        num_predictors=1
    elif(model_type == "quadratic" or model_type == "NN"):
        num_params=6 
        num_predictors=2
    else:
        print("Not implemented model of type", model_type)
        exit()

    summary_method_param = "raw"                                        #"raw", "moments", "percentiles" or "JSD"
    if(summary_method_param == "JSD"):
        distance_type = "sqrt_JSD"
    else:
        distance_type = "rmse"
    
    #Creating the prior, using sp.stats. If we need numba pdf evaluation this is not the best choice:
    prior_means, prior_covs, noise_std = create_gaussian_prior(model_type)
    prior_distr=sp.stats.multivariate_normal(mean=prior_means, cov= prior_covs)

    print("Running algorithm with prior_size=%d, percentile_threshold=%.3f, posterior_size=%d, MCMC_posterior_size=%d, noise_std=%.3f, num_observations=%d,"
    " \npost_process_flag=%d, summary_method=%s, model_type=%s\n" % (num_prior_samples, percentile_threshold, num_posterior_samples, MCMC_posterior_size, noise_std, N, post_process_flag, summary_method_param, model_type))
    print("prior mean", prior_means)
    print("prior cov\n", prior_covs)
    #FINISHED WITH THE INITIALISATION#

    #GETTING GROUND TRUTH, EVIDENCE AND THE TARGET POSTERIOR (EITHER THEORETICAL OR MCMC INFERRED)#
    random_state_used=42                   #In case we want reproductibility 

    ground_truth = prior_distr.rvs(size=1, random_state=random_state_used)         
    X_obs=get_random_X(N=N, num_predictors=num_predictors, random_state=random_state_used)     
    Y_obs,_=simulate_output_ABC(X_obs, ground_truth, noise_std, model_type, random_state=random_state_used)
    plt.hist(Y_obs, bins=20, edgecolor='black')
    plt.title("Histogram of observed Y")


    print("\nthe ground truth is ", ground_truth)
    print("X obs final element", X_obs[-1])
    print("Y obs final element", Y_obs[-1])

    # print("final y obs:", Y_obs[-1])
    # print("ground truth", ground_truth)
    # print("final x obs", X_obs[-1])
    if(model_type =="linear" or model_type=="quadratic"):
        posterior_means,posterior_covs=get_theoretical_posterior(prior_means, prior_covs, noise_std, X_obs, Y_obs, model_type)  
        # MCMC_posterior, MCMC_means, MCMC_cov=mcmc_inference(X_obs, Y_obs, MCMC_posterior_size, prior_means, prior_covs, model_type, noise_std, progress_bar=True)   
    
    if(model_type == "NN"):
        path_to_file = "MCMC_posteriors/N=" + str(N) + ",noise=" + str(noise_std) + ".npy"
        if os.path.exists(path_to_file):
            print("\nLoading MCMC posterior from File with path ",path_to_file)
            MCMC_posterior = np.load(path_to_file)
            posterior_means = np.mean(MCMC_posterior, axis=0)         
            posterior_covs = np.cov(MCMC_posterior, rowvar=False, bias=True)  
            print("Posterior means:", posterior_means)
            print("Posterior covariance:\n", posterior_covs) 
            
        else: 
            print("File with path ",path_to_file, "Not found, please offer path containing MCMC posterior")
            exit()
    JSD_bin_edges = get_bin_edges(Y_obs, "FD")                           #These are only used when we calculate the JSD 
    #FINISHED WITH EVIDENCE, TARGET POSTERIOR#

    #APPLYING THE ABC LOOP#
    t3=time.time()
    ABC_empirical_samples, ABC_mean, ABC_cov, final_distances, final_seeds = ABC_loop(prior_distr, num_prior_samples, X_obs, Y_obs, noise_std, summary_method_param, model_type, distance_type, num_posterior_samples, JSD_bin_edges)
    epsilon_achieved = np.max(final_distances)

    print("\nABC results:")
    print("Means:", ABC_mean)
    print("Covs:\n", ABC_cov)
    print("epsilon achieved", epsilon_achieved)

    t2=time.time()
    print("ABC process time: ",t2-t3)
    #FINISHED WITH THE ABC LOOP#

    #APPLYING THE POST PROCESSING CORRECTION#
    if(post_process_flag):
        corrected_samples, corrected_empirical_mean, corrected_empirical_cov = post_processing_ABC(ABC_empirical_samples, final_distances, final_seeds, summary_method_param, model_type, distance_type, Y_obs, X_obs, noise_std, JSD_bin_edges)
        print("Corrected results:")
        print("Means:", corrected_empirical_mean)
        print("Covs:\n", corrected_empirical_cov)
        KL_corrected = kl_divergence_gaussians(posterior_means, posterior_covs, corrected_empirical_mean, corrected_empirical_cov)
    else:
        corrected_samples = None ; corrected_empirical_mean = None; corrected_empirical_cov = None; KL_corrected = None
    #FINISHED WITH THE POST PROCESSING CORRECTION#

    #PRINTING AND PLOTTING RESULTS#
    KL_ABC = kl_divergence_gaussians(posterior_means, posterior_covs, ABC_mean, ABC_cov)
    print("\nKL ABC=", KL_ABC, "KL_corrected=", KL_corrected)
    #FINISHED PRINTING AND PLOTTING RESULTS#
    
    if(model_type == "linear" or model_type == "quadratic"):
        plot_gaussian_distributions(prior_means, prior_covs, posterior_means, posterior_covs, ABC_mean, ABC_cov, corrected_empirical_mean, corrected_empirical_cov, ground_truth, num_rows=int(num_params/2) , num_cols=2)
    
    if(model_type == "NN"):
        plot_non_gaussian_distributions(prior_means, prior_covs, MCMC_posterior, ABC_empirical_samples, corrected_samples, ground_truth, num_rows=int(num_params/2) , num_cols=2)

    # plt.show()


if __name__ == "__main__":
    main() 

