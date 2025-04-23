#Implementation of Del Morals Adaptive SMC ABC Algorithm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
from library import create_gaussian_prior, get_random_X, generate_noise_vector, linear_model, quadratic_model, NN_model, get_theoretical_posterior, get_bin_edges, get_summary_statistics, kl_divergence_gaussians, distance_metric, simulate_output_ABC, linear_model_numba
from library_present import plot_gaussian_distributions, plot_non_gaussian_distributions, plot_history
import sys
import time  
import os 
import torch
from numba import njit
import threading

#Initialising our population (i.e. getting initial particles and weights)
#Done by uniform sampling from the prior and assigning equal weights everywhere
def initialise_population(pop_size, prior_distr):
    particles=prior_distr.rvs(size=pop_size)        
    weights=np.ones(shape=(len(particles)))/len(particles)
    return particles, weights

#Get means and covariances from the particles,weights pair
def get_empirical_moments(particles, weights):
    weights = weights / np.sum(weights)
    mean = np.sum(particles * weights[:, np.newaxis], axis=0)
    cov = np.cov(particles, rowvar=False, aweights=weights, bias=True)
    
    return mean, cov

#Computes ESS given weights
def get_ESS(weights):
    X=np.sum(weights**2)
    return 1/X
  
#This function implements our resample criterion
#Initially if ESS<threshold -> True, can be changed 
def resample_criterion(ESS, threshold):
    if(ESS<threshold):
        return True
    else:
        return False

#Implementing equation 14 of Del Morals Adaptive SMC ABC 
def compute_weights(previous_weights, distances, previous_epsilon, current_epsilon):

    denominators = np.sum(distances<previous_epsilon, axis=1)
    numerators = np.sum(distances<current_epsilon, axis=1)

    fractions = np.zeros(shape=(len(numerators)))
    valid_indices = denominators > 0                #to avoid division by zero
    fractions[valid_indices] = numerators[valid_indices] / denominators[valid_indices]

    new_weights= previous_weights * fractions

    new_weights = new_weights/np.sum(new_weights)
    return new_weights

#Implementing step 1 of del morals algorithm
def compute_threshold(previous_weights, distances, previous_epsilon, previous_ESS, alpha):

    #This solves using the bisection solution. Essentially we want to find an epsilon value that corresponds to a weight set that has the 
    #desired ESS, given by the adaptive rule of Del Moral.
    #Done by searching root for f(epsilon) = ESS(new_weights) - alpha * previous_ESS
    def bisection_solution(previous_weights, distances, previous_epsilon, previous_ESS, alpha, tol=1e-8, max_iter=100000):
        
        # Define the function f(epsilon) = ESS(new_weights) - alpha * previous_ESS
        def f(epsilon):
            # Compute new weights using the provided compute_weights function
            new_w = compute_weights(previous_weights, distances, previous_epsilon, epsilon)
            return get_ESS(new_w) - alpha * previous_ESS

        
        eps_low = 1e-6
        eps_high = previous_epsilon

        
        # Bisection loop. I use tol as tolerance both for how close the epsilon values of the left-right points are, but also 
        # how close the f is to 0.
        iter_count = 0
        while (eps_high - eps_low) > tol and iter_count < max_iter:
            
            eps_mid = (eps_low + eps_high) / 2.0
            f_mid = f(eps_mid)
            
            if abs(f_mid) < tol:            #Solved the equation
                current_weights=compute_weights(previous_weights, distances, previous_epsilon, eps_mid)
                current_ESS = get_ESS(current_weights)
                return current_weights, current_ESS,eps_mid
            
            if f_mid > 0:
                eps_high = eps_mid
            else:
                eps_low = eps_mid
            
            iter_count += 1

        epsilon_n = (eps_low + eps_high) / 2.0

        new_weights = compute_weights(previous_weights, distances, previous_epsilon, epsilon_n)
        ESS = get_ESS(new_weights)
        return new_weights, ESS, epsilon_n
    
    return bisection_solution(previous_weights, distances, previous_epsilon, previous_ESS, alpha)

#Implement resampling. Both multinomial (simple) and stratified (said to lead to reduced weight variance)
#In Multinomial we pick uniformly from [0,1] and then use this position to inverse the CDF that is derived from the weights. The solution is the particle index
#In stratified we create N equally sized bins (Strata) in [0,1] and pick uniformly from each one as positions - 1 position per stratum.
#This way we have better coverage but still respect the CDF.
def resample_empirical(particles, weights, distances, summaries_Y, method="stratified"):

    if(method == "multinomial"):
        #Implement Multinomial Resampling
        N = len(particles)
        new_element_indices = np.random.choice(N, size=N, p=weights)          #numpy to do multinomial resampling!
        new_particles =    particles[new_element_indices]
        new_distances =    distances[new_element_indices]
        new_summaries_Y=   summaries_Y[new_element_indices]
        new_weights = np.ones(shape=(N))/N

    if(method == "stratified"):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)                             #The cdf
        positions = (np.arange(N) + np.random.rand(N)) / N              #The positions. I get strata (np.arange(N))/N and pick uniformly in each one
        new_element_indices = np.searchsorted(cumulative_sum, positions)        #inversing 
        new_particles = particles[new_element_indices]
        new_distances = distances[new_element_indices]
        new_summaries_Y = summaries_Y[new_element_indices]
        new_weights = np.ones(N) / N
    

    return new_particles, new_weights, new_distances , new_summaries_Y 

#Post processing SMC ABC. Here we do not use the seeds to recreate the added noises, we carry the whole summary statistic information.
#We also have to do adjustments so that it can work with SMC ABC scheme. First of all resampling so that we have an essentially unweighted empirical distribution
#Second we have to somehow solve the problem associated with the M>1 datasets. 
def post_processing_SMC_ABC(current_particles, current_dists, current_weights, current_summaries_Y, X_obs, Y_obs, summary_method_param, resampling_method, JSD_bin_edges):

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
    
    #This function takes the result of Del Moral's algorithm, where M>1. 
    #For each particle, it accepts one simulated dataset and the corresponding summary statistic set.
    #The way it is done is by picking the one out of M simulated datasets with the smallest distance.
    def arrays_for_post_processing(current_particles, current_dists, current_summaries_Y):
        pop_size            = len(current_particles)
        summary_set_length = len(current_summaries_Y[0][0])
        return_dists = np.zeros(shape=(pop_size, ))
        return_summaries = np.zeros(shape=(pop_size, summary_set_length))

        for index in range(len(current_particles)):
            dists=current_dists[index]
            best_dist_pos = np.argmin(dists)
            best_dist = dists[best_dist_pos]
            corresponding_summary = current_summaries_Y[index][best_dist_pos]

            return_dists[index] = best_dist
            return_summaries[index] = corresponding_summary
        
        return current_particles, return_dists, return_summaries

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
   
    #Resampling if needed, because the method works with empirical distributions without weights (i.e. uniform weights)
    if(np.any(current_weights != current_weights[0])):
        current_particles, current_weights, current_dists, current_summaries_Y = resample_empirical(current_particles, current_weights, current_dists, current_summaries_Y, resampling_method)
    summary_Y_obs = get_summary_statistics(Y_obs, summary_method_param, JSD_bin_edges)

    #Applying the correction
    posterior_samples, distances, summaries = arrays_for_post_processing(current_particles, current_dists, current_summaries_Y)
    print("Post processing Information:")
    print("max distances:", np.max(distances), "min distances:", np.min(distances), "std of distances: ", np.std(distances))
    Q, X, B_hat=solve_regression(posterior_samples, distances, summaries, summary_Y_obs)
    calculate_R2(posterior_samples, X, B_hat)

    corrected_samples = posterior_samples - (summaries - summary_Y_obs) @ Q

    empirical_mean = np.mean(corrected_samples, axis=0)
    empirical_cov = np.cov(corrected_samples, rowvar=False)
    t2=time.time()
    print("Time for correction", t2-t1)
    print("corrected Means", empirical_mean)
    print("corrected covs", empirical_cov)
    return corrected_samples, empirical_mean, empirical_cov
 

 #Function for efficient sampling from gaussian distribution!!


#The perturbation kernel is gaussian, with a covariance of (cov_factor * empirical_covariance of the previous gen).
#I need to sample from this multivariate gaussian and add as mean some particle. To do this effectively i use the cholesky decomposition.
#I compute kernel cholesky L (covariance_matrix = L*L.T where L is a lower triangular matrix). To sample, I sample from the sample~N(0, I) and apply the 
#transform  particle + L @ sampled_value. this makes the sampling efficient.
@njit
def fast_multivariate_normal(mean, L):          #Implements the random pertrurbation, using cholesky factor estimated previously!
    z = np.random.standard_normal(size=mean.shape[0])           #sample from standard normal
    # Transform it using the Cholesky factor
    return mean + L @ z


@njit
def get_MH_probability(prior_means, prior_cov_inv, prior_log_norm_const, init_particle, perturbed_particle, init_distances, perturbed_distances, epsilon):
    
    #Use this to get gaussian pdf.
    #It works by computing the logp(x) = -1/2 * (x-μ)T * Σ^-(1) * (x-μ) - d/2 * log(2π) - 1/2 * log_det(Σ). Then taking the exp() for the pdf p(x).
    #The term d/2 * log(2π) - 1/2 * log_det(Σ) is the log_norm_const already evaluated and passed as argument
    #This could further be optimising by just evaluating log(p(x1)) - log(p(x2)) but there is no need to for the time being.
    def fast_gaussian_pdf(particle, prior_means, cov_inv, log_norm_const):
        diff = particle - prior_means
        exponent = -0.5 * diff @ cov_inv @ diff
        return np.exp(log_norm_const + exponent)
    
    perturbed_num_acc =  np.sum(perturbed_distances<=epsilon)
    init_num_acc =   np.sum(init_distances<=epsilon)
    perturbed_prior_pdf = fast_gaussian_pdf(perturbed_particle, prior_means, prior_cov_inv, prior_log_norm_const)
    init_prior_pdf = fast_gaussian_pdf(init_particle, prior_means, prior_cov_inv, prior_log_norm_const)
    if(init_prior_pdf ==0):
        raise ValueError("Init prior pdf is 0")
    if(init_num_acc ==0):
        raise ValueError("denominator is 0 in get mh probability")

    MH_prob=perturbed_prior_pdf * perturbed_num_acc / (init_prior_pdf * init_num_acc)
    return min(1, MH_prob)
    

#This function is similar as before, but now the main computation takes place inside the worker_fun, which is the function run by the threads
def get_new_particles(current_weights, current_particles, current_distances, current_summaries_Y, prior_means, prior_covs, X_obs, Y_obs, noise_std, epsilon, M, summary_method_param, JSD_bin_edges, distance_type, model_type, NUM_THREADS, cov_factor=2, epoch=None):

    #We want to precompute the log_norm_const = - d/2 * log(2π) - 1/2 * log_det(Σ) and the prior_conv_inv = Σ^-(1), so we can evaluate the 
    #log(p(x)) fast. Check the get_MH_probability function to understand.
    def precompute_gaussian_terms(prior_covs):
        prior_cov_inv = np.linalg.inv(prior_covs)
        _, prior_cov_logdet = np.linalg.slogdet(prior_covs)
        dim = prior_covs.shape[0]
        
        prior_log_norm_const = -0.5 * (dim * np.log(2 * np.pi) + prior_cov_logdet)
        return prior_cov_inv, prior_log_norm_const

    def worker_fun(current_particles, current_weights, current_distances, current_summaries_Y, next_particles, next_distances, next_summaries_Y, kernel_cholesky, N, M, noise_std, X2d,
                    Y_obs, summary_method_param, JSD_bin_edges, distance_type, noise_matrix, access_indices, model_code, epsilon, prior_means, prior_cov_inv, prior_log_norm_const,
                    population_access_lock, fetching_lock, simulation_times_lock, regen_lock, regen_started, regen_done, termination_array, thread_id, integer_dictionary):
        
        access_indices_position = 0
        pop_size = len(current_particles)
        simulation_times_called = 0

        while(1):

            #1) Picking a population index to work with using mutual exclusion:
            #If the index exceeds the population size, this means that the thread essentially has finished. once this happens, the thread sets termination_array[thread_id] = 1
            #if all the threads are finished, then the thread increases the simulation_times_called through mutual exclusion with the simulation_times_lock and returns.
            with population_access_lock:
                this_index = integer_dictionary["population_access_index"]
                if(this_index >= pop_size):
                    termination_array[thread_id] = 1
                if(np.sum(termination_array) == len(termination_array)):
                    with simulation_times_lock:                 #Also updating the amount of times we simulated
                        print("Thread id ", thread_id, "finished with simulation times called", simulation_times_called)
                        integer_dictionary["simulation_times_called"] +=simulation_times_called
                    return
                
                integer_dictionary["population_access_index"] +=1

            #2) Doing work with this index until i have to fetch a noise value:
            #If the thread is not finished and the particle is dead, carry it to the next generation and start the loop again (continue command)
            if(this_index < pop_size):
                this_weight = current_weights[this_index]
                this_particle = current_particles[this_index]
                if(this_weight==0):             #dead particle, no need to try to simulate
                    next_particles[this_index]    = this_particle
                    next_distances[this_index]    = current_distances[this_index]
                    next_summaries_Y[this_index]  = current_summaries_Y[this_index]
                    continue
            
            while(1):

                new_perturbed_particle = fast_multivariate_normal(this_particle, kernel_cholesky)

                #3) Now fetching the noise value and perhaps creating a new noise matrix. 
                #At first fetching a noise block. This happens only if there are noise blocks left. If there are now blocks left, need_regen is set to true and we must regenerate
                with fetching_lock:
                    access_indices_position = integer_dictionary["noise_matrix_access_index"]
                    if(access_indices_position >= len(access_indices)):                   #This means that we have fetched all the blocks of the noise matrix
                        need_regen = True
                    else:                                                               #This means that we can fetch the element
                        integer_dictionary["noise_matrix_access_index"] = access_indices_position + 1
                        noise_block_pointer = access_indices[access_indices_position]
                        need_regen = False

                #Now regenerating:
                #The first thread that reaches the regeneration lock, sets the regen_started event and becomes the leader thread.
                #It then generates the matrices and saves them in place (shared memory). 
                #It also signals that the regeneration is over, when it is not the threads wait, so that they are synchronised when the regeneration ends.
                #We then also clear the events for the next iterations.
                if need_regen:
                    with regen_lock:
                        if not regen_started.is_set():
                            regen_started.set()  # I’m the leader

                            noise_matrix[:], access_indices[:] = create_noise_matrix(N, M, noise_std)
                            integer_dictionary["noise_matrix_access_index"] = 0

                            regen_done.set()  

                    regen_done.wait()  # all threads wait here

                    with regen_lock:
                        regen_started.clear()
                        regen_done.clear()

                    continue

                #4) Doing the rest of the work:
                #If the thread is not finished, simulate and compute MH probability as usual. If it is finished, break and return to the beginning to test whether to return.
                if(this_index < pop_size):
                    simulation_times_called += 1
                    new_distances,new_num_accepted, new_summaries_Y = simulate_particle(new_perturbed_particle, X2d, Y_obs, M, summary_method_param, JSD_bin_edges, distance_type, noise_matrix, noise_block_pointer, model_code, epsilon)
                    new_prob_acceptance = get_MH_probability(prior_means, prior_cov_inv, prior_log_norm_const, this_particle, new_perturbed_particle, current_distances[this_index], new_distances, epsilon)

                    if(np.random.rand() <= new_prob_acceptance): 
                        next_particles[this_index]    = new_perturbed_particle
                        next_distances[this_index]    = new_distances
                        next_summaries_Y[this_index]  = new_summaries_Y
                        break 
                
                else:
                    break



    if(epoch>-1):                   #This can help me understand if i lost diversity... 
        print("Unique elements:")
        print(len(np.unique(current_particles, axis=0)))


    model_code = 0 if model_type == "linear" else 1 if model_type == "quadratic" else 2 if model_type == "NN" else -1       #Also defining the model_code concept, because in njit functions we cannot have if(... == "string") and it has to be implemented with integers
    N = len(Y_obs)
    X2d = X_obs if X_obs.ndim == 2 else X_obs[:, None].repeat(2, axis=1)


    #Computing cholesky decomposition for kernel covariance matrix -> fast multivariate gaussian sampling to get perturbation
    #Also getting the constant terms of the log(p(x)) computation, to have fast pdf evaluation of prior, needed for MH probability. 
    kernel_cov = cov_factor * np.cov(current_particles, rowvar=False, aweights=current_weights, bias=True)
    kernel_cholesky = np.linalg.cholesky(kernel_cov)            
    prior_cov_inv, prior_log_norm_const = precompute_gaussian_terms(prior_covs)

    #For the Y summaries, i need length of the summary set 
    summary_y_obs = get_summary_statistics(Y_obs, summary_method_param, JSD_bin_edges)
    next_summaries_Y=np.zeros(shape=(len(current_weights), M, len(summary_y_obs)))

    
    next_particles=np.zeros_like(current_particles)
    next_distances=np.zeros(shape=(len(current_weights), M))
    

    #In each SMC ABC trial i initialise a noise_matrix. Each access index points to the first element of a (M,) block. Each simulate_particle call used on of these blocks.
    #Implemented this way so that it can be easily parallelisable (noise_matrix in shared memory, each thread picks on access_index)
    noise_matrix, access_indices = create_noise_matrix(N, M, noise_std)
    

    #Setting up the parallel algorithm:
    population_access_lock = threading.Lock()                               #Lock for mutual exclusion when picking particle for the next population
    fetching_lock = threading.Lock()                                        #Lock for mutual exclusion when fetching a value from the noise matrix
    simulation_times_lock = threading.Lock()                                #Lock for mutual exclusion when setting the amount of times we used simulate_particle
    regen_lock = threading.Lock()                                           #Lock for mutual exclusion when regenarating a new noise matrix

    regen_started = threading.Event()                                       #Event for the start of the regeneration process
    regen_done = threading.Event()                                          #Even for the termination of the regeneration process


    integer_dictionary = {}                                                 #Dictionary to easily handle integer values in shared memory (essentially to store them in heap)
    integer_dictionary["simulation_times_called"] = 0
    integer_dictionary["population_access_index"] = 0
    integer_dictionary["noise_matrix_access_index"] = 0

    termination_array = np.zeros(shape=(NUM_THREADS,))                      #Array that has 1 on the ith position if the ith thread has finished. If all are =1 then we can return the threads.
    thread_list =[]                 #My thread list
    for thread_id in range(NUM_THREADS):
        current_thread = threading.Thread(target=worker_fun, args=(current_particles, current_weights, current_distances, current_summaries_Y, next_particles, next_distances, next_summaries_Y, kernel_cholesky, N, M, noise_std, X2d,
                    Y_obs, summary_method_param, JSD_bin_edges, distance_type, noise_matrix, access_indices, model_code, epsilon, prior_means, prior_cov_inv, prior_log_norm_const,
                    population_access_lock, fetching_lock, simulation_times_lock, regen_lock, regen_started, regen_done, termination_array, thread_id, integer_dictionary))
        current_thread.start()
        thread_list.append(current_thread)

    #Waiting for them to finish
    for t in thread_list:
        t.join()

    simulation_times_called = integer_dictionary["simulation_times_called"] 
    print("Simulation Times Called", simulation_times_called)
    
    return next_particles, next_distances, next_summaries_Y, simulation_times_called

#This function is changed in the parallel version, it is now njitted.
#To make sure that the X_obs input is always of the same shape (not 1D for Linear and 2D for The others) which is something numba wants, I pass as X2d
#Then, the linear_model_numba handles the 2D and gets the first column.
@njit
def simulate_particle(particle, X2d, Y_obs, M, summary_method_param, JSD_bin_edges, distance_type, noise_matrix, noise_matrix_index, model_code, epsilon=None):
    
    distances=np.zeros(shape=(M,))
    summary_Y_obs = get_summary_statistics(Y_obs, summary_method_param, JSD_bin_edges)
    summaries_Y = np.zeros(shape=(M, len(summary_Y_obs)))
    accepted_dist_count=0

    if(model_code == 0):
        non_noisy_Y = linear_model_numba(X2d , particle)
    elif(model_code == 1):
        non_noisy_Y = quadratic_model(X2d, particle)
    elif(model_code == 2):
        non_noisy_Y = NN_model(X2d, particle)

    #We have to do M simulations for the given particle. We do the forward pass once and then just add a noise instance. 
    for i in range(M):
        current_noise = noise_matrix[noise_matrix_index]
        current_Y = non_noisy_Y + current_noise
        noise_matrix_index +=1
        summary_current_Y = get_summary_statistics(current_Y, summary_method_param, JSD_bin_edges)
        current_metric=distance_metric(summary_Y_obs, summary_current_Y, distance_type)
        distances[i]=current_metric
        summaries_Y[i]=summary_current_Y
        if(epsilon is not None):
            if(current_metric<=epsilon):
                accepted_dist_count+=1
    return distances, accepted_dist_count, summaries_Y


def create_noise_matrix(N, M, noise_std, GBs=1):

    #First estimating the number of rows 
    #num_rows x num_cols x num_bytes(float64) = 1024**3*GBs , through this equation we get the num_rows
    #we also multiply by M afterwards to make the result a multiple of M. This is done to make the process easy to use when wanting to generate M noise instances per particle.
    #Of course we also divide by M before getting the integer.
    num_rows = int(GBs*1024**3/(N * 8 * M)) * M

    # #Now initialising and filling up the noise matrix
    # noise_matrix = np.zeros(shape=(num_rows, N), dtype=np.float64)
    # for current_row in range(num_rows):
    #     noise_matrix[current_row] = get_random_noise(noise_std, N)
    
    if torch.cuda.is_available():
        noise_matrix = torch.normal(0.0, noise_std, size=(num_rows, N), dtype=torch.float64, device="cuda").cpu().numpy()
    else:
        noise_matrix = torch.normal(0.0, noise_std, size=(num_rows, N), dtype=torch.float64, device="cpu").numpy()
    
    access_indices = np.arange(0, num_rows, M)              #Each access index is the position of the first element in one (M,) block. Passed to one simulate_particle call
    return noise_matrix, access_indices


@njit 
def get_random_noise(noise_std, N):
    return np.random.normal(0, noise_std, N)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------


#Global Variables:
def main():

    t_init=time.time()
    #HYPERPARAMETERS OF THE ALGORITHM#
    pop_size=500                                #Pop Size
    M=5                                           #Amount of times we simulated each parameter
    alpha=0.9                                   #Parameter that helps with the threshold adaptivity algorithm
    NT=pop_size/2                                      #Resampling threshold (If ESS<NT resample). 
    epsilon = 1                              #If et < epsilon stop 
    cov_factor = 2                               #Helps us get the covariance of the gaussian perturbation kernel. cov_factor * previous_empirical_variance  
    cov_reduction_factor = 0.995                #Helps with adaptive cov_factor. 1 if we do not want adaptation. 0.995 good choice
    resampling_method="stratified"              #"stratified" or "multinomial" or "systematic"
    N=1000                                       #Number of observations
    post_process_flag=0
    smc_time_limit = 60 * 60
    NUM_THREADS = 2


    model_type="quadratic"                                                 #"linear" "quadratic" or "NN"
    if(model_type == "linear"):
        num_params=2
        num_predictors=1
    elif(model_type =="quadratic" or model_type =="NN"):
        num_params=6 
        num_predictors=2
    else:
        print("Not implemented model of type", model_type)
        exit()
    model_code = 0 if model_type == "linear" else 1 if model_type == "quadratic" else 2 if model_type == "NN" else -1       #Also defining the model_code concept, because in njit functions we cannot have if(... == "string") and it has to be implemented with integers


    summary_method_param = "raw"                                        #"raw", "moments", "percentiles" or "JSD"
    if(summary_method_param == "JSD"):
        distance_type = "sqrt_JSD"
    else:
        distance_type = "rmse"

    prior_means, prior_covs, noise_std = create_gaussian_prior(model_type)
    prior_distr=sp.stats.multivariate_normal(mean=prior_means, cov= prior_covs)
    print("Running SMC ABC algorithm with pop_size=%d, M=%d, alpha=%.3f, Resampling Threshold=%d, epsilon=%.3f, cov_factor=%.3f, cov_reduction_factor=%.3f noise_std=%.3f, observations_per_dataset=%d, resampling_method=%s, post_process_flag=%d, summary_method_param=%s, SMC_time_limit=%d, model_type=%s, NUM_THREADS=%d"
           % (pop_size, M, alpha, NT, epsilon, cov_factor, cov_reduction_factor, noise_std, N, resampling_method, post_process_flag, summary_method_param, smc_time_limit, model_type, NUM_THREADS))
    print("prior mean", prior_means)
    print("prior cov\n", prior_covs)
    #FINISHED WITH THE ALGORITHM HYPERPARAMETERS#

    #GETTING THE GROUND TRUTH - EVIDENCE, THE THEORETICAL POSTERIOR OR THE MCMC RESULT#
    random_state_used=42        #In case we want reproductibility 
    torch.manual_seed(random_state_used)
    ground_truth = prior_distr.rvs(size=1, random_state=random_state_used)         #Delete random seed if reproductibility is not needed 
    X_obs=get_random_X(N=N, num_predictors=num_predictors, random_state=random_state_used) 
    X2d = X_obs if X_obs.ndim == 2 else X_obs[:, None].repeat(2, axis=1)

    Y_obs, _ = simulate_output_ABC(X_obs, ground_truth, noise_std, model_type, random_state=random_state_used)
    plt.hist(Y_obs, bins=int(np.sqrt(N)), edgecolor='black')
    plt.title("Histogram of observed Y, N=" + str(N))


    print("the ground truth is ", ground_truth)
    print("X obs final element", X_obs[-1])
    print("Y obs final element", Y_obs[-1])

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
            print("Posterior covariance:\n", posterior_covs, "\n") 
            
        else: 
            print("File with path ",path_to_file, "Not found, please offer path containing MCMC posterior")
            exit()
    JSD_bin_edges = get_bin_edges(Y_obs, "SR")                           #These are only used when we calculate the JSD 
    #FINISHED THE GROUND TRUTH - EVIDENCE, THE THEORETICAL POSTERIOR AND THE MCMC RESULT#

    #INITIALISING POPULATION#
    #populations_list[i] is a dictionary with keys "particles", "weights", "distances", "means", "covs", "ess", "epsilon", "cov_factor", "kl_divergence"
    #"particles" pop_size x num_params, "weights" pop_size , "distances" pop_size x M, "means" num_params, "covs" num_params x num_params, "ess" float, "cov_factor" float. "kl_divergence" float
    print("Initialising the population")
    t_SMC_init = time.time()
    populations_list=[]

    #Now getting the initial population and appending in the populations list. 
    Parts0, Weights0=initialise_population(pop_size, prior_distr)
    dists0 = np.zeros(shape=(pop_size, M))
    current_summaries_Y = np.zeros(shape=(pop_size, M, len(get_summary_statistics(Y_obs, summary_method_param, JSD_bin_edges))))             #I cant keep up with logging this due to too much memory in cse the summary set is not small 
    
    noise_matrix, access_indices = create_noise_matrix(N, M, noise_std)
    access_indices_position = 0
    for i in range(len(Parts0)):
        if(access_indices_position >= len(access_indices)):         #This means we have to get a new matrix
            noise_matrix, access_indices = create_noise_matrix(N, M, noise_std)
            access_indices_position = 0 

        current_dists,_,summaries_Y = simulate_particle(Parts0[i], X2d, Y_obs, M, summary_method_param, JSD_bin_edges, distance_type, noise_matrix, access_indices[access_indices_position], model_code, None)
        access_indices_position +=1 
        dists0[i] = current_dists
        current_summaries_Y[i] = summaries_Y
    epsilon0=np.max(dists0)
    empirical_means0,empirical_covs0 = get_empirical_moments(Parts0, Weights0)
    kl_divergence0 = kl_divergence_gaussians(posterior_means, posterior_covs, empirical_means0, empirical_covs0)
    ess0=get_ESS(Weights0)

    pop0={"particles": Parts0, "weights": Weights0, "distances": dists0, "means":empirical_means0, "covs":empirical_covs0, "ess": ess0, "epsilon": epsilon0, "cov_factor":cov_factor, "kl_divergence": kl_divergence0}
    populations_list.append(pop0)
    print("Initial means:", empirical_means0)
    print("Initial covs:", empirical_covs0)

    
    #Extra logging information (secondary):
    current_gen_time=[]                         #Logging total time per generation
    resampling_generation=[]            #List holding all generation when we did resampling
    num_simulations_per_generation=[]
    #These are logged only when we use post processing correction:
    corrected_means_list = []
    corrected_covs_list  = []
    corrected_kl_divergence_list =[]
    #FINISHED INIITIALISING POPULATION#


    #RUNNING SMC ABC LOOP#
    print("Starting the SMC ABC Loop")
    current_epsilon = epsilon0
    current_weights = Weights0
    current_ESS = ess0
    current_particles = Parts0
    current_dists = dists0
    current_kl_divergence = kl_divergence0
    gen_num=0
    t4=time.time()                  #t1-> Before Step 1, t2-> Between Step 1 and Step 2, t3-> Right before step 3, t4-> Right After Step 3
    while(1):

        gen_num +=1 

        #Step 1: calculating the new weights and the new ESS and epsilon
        t1=time.time()
        current_weights, current_ESS, current_epsilon = compute_threshold(current_weights, current_dists, current_epsilon, current_ESS, alpha)
        t2=time.time()

        #Step 2: Resampling if needed, getting new statistics, pushing population in logging list
        if(resample_criterion(current_ESS, NT)):
            resampling_generation.append(gen_num)
            current_particles, current_weights, current_dists, current_summaries_Y = resample_empirical(current_particles, current_weights, current_dists, current_summaries_Y, resampling_method)
            current_ESS = get_ESS(current_weights)
        current_means, current_covs = get_empirical_moments(current_particles, current_weights)
        current_kl_divergence = kl_divergence_gaussians(posterior_means, posterior_covs, current_means, current_covs)
        current_pop={"particles": current_particles, "weights": current_weights, "distances": current_dists, "means": current_means, 
                      "covs": current_covs, "ess": current_ESS, "epsilon": current_epsilon, "cov_factor": cov_factor, "kl_divergence": current_kl_divergence}
        populations_list.append(current_pop)
        print("Generation=%d, ess=%.3f, epsilon=%.3f" % (gen_num, current_ESS, current_epsilon))
        print("Means", current_means)
        print("covs", current_covs)
        print("KL Divergence", current_kl_divergence, "\n")
        if(post_process_flag == 1):
            current_corrected_samples, current_corrected_means, current_corrected_covs = post_processing_SMC_ABC(current_particles, current_dists, current_weights, current_summaries_Y, X_obs, Y_obs, summary_method_param, resampling_method, JSD_bin_edges)
            corrected_kl_divergence = kl_divergence_gaussians(posterior_means, posterior_covs, current_corrected_means, current_corrected_covs)
            corrected_means_list.append(current_corrected_means)
            corrected_covs_list.append(current_corrected_covs)
            corrected_kl_divergence_list.append(corrected_kl_divergence)
            print("Corrected KL Divergence", corrected_kl_divergence, "\n")
        
        #Checking if termination criteria are satisfied:
        if(current_epsilon<=epsilon):
            print("\n\nSATISFIED EPSILON GOAL RESULT! Generation", gen_num, "!\n")
            break
        
        if(t4 - t_SMC_init > smc_time_limit):
            print("\n\nTIME LIMIT REACHED! Generation", gen_num, "!\n")
            break


        #Step 3: Create new population
        t3=time.time()
        
        cov_factor = cov_factor * cov_reduction_factor
        current_particles, current_dists, current_summaries_Y, current_num_perturbs = get_new_particles(current_weights, current_particles, current_dists, current_summaries_Y,
                                                                                                        prior_means, prior_covs, X_obs, Y_obs, noise_std, current_epsilon, M, summary_method_param,
                                                                                                        JSD_bin_edges, distance_type, model_type, NUM_THREADS, cov_factor, gen_num) 
        num_simulations_per_generation.append(current_num_perturbs)

        t4=time.time()
        current_gen_time.append(t4-t1)
        print("total time till now", t4-t_init)
        print("Step 1=%.3f Step 2=%.3f Step 3=%.3f" %(t2-t1, t3-t2, t4-t3))


    #Final Resample if needed:
    if(np.any(current_weights != current_weights[0])):
      current_particles, current_weights, current_dists, current_summaries_Y = resample_empirical(current_particles, current_weights, current_dists, current_summaries_Y, resampling_method)
    current_means, current_covs = get_empirical_moments(current_particles, current_weights)
    current_kl_divergence = kl_divergence_gaussians(posterior_means, posterior_covs, current_means, current_covs)

    #Final Correction:
    if(post_process_flag ==1):
        current_corrected_samples, current_corrected_means,current_corrected_covs = post_processing_SMC_ABC(current_particles, current_dists, current_weights, current_summaries_Y, X_obs, Y_obs, summary_method_param, resampling_method, JSD_bin_edges)
        corrected_kl_divergence = kl_divergence_gaussians(posterior_means, posterior_covs, current_corrected_means, current_corrected_covs)
        
        corrected_means_list.append(current_corrected_means)
        corrected_covs_list.append(current_corrected_covs)
        corrected_kl_divergence_list.append(corrected_kl_divergence)
        print("Final Corrected KL Divergence", corrected_kl_divergence)
    else:                           
        current_corrected_samples = None; current_corrected_means = None; current_corrected_covs = None; corrected_kl_divergence = None; 
        corrected_means_list = None; corrected_covs_list = None; corrected_kl_divergence_list = None 
    print("Final KL Divergence:", current_kl_divergence)
    #FINISHED RUNNING SMC ABC LOOP#

    #PRESENTING THE RESULTS#
    t_final=time.time()
    print("Finished running SMC ABC algorithm with pop_size=%d, M=%d, alpha=%.3f, Resampling Threshold=%d, epsilon=%.3f, cov_factor=%.3f, cov_reduction_factor=%.3f noise_std=%.3f, observations_per_dataset=%d, resampling_method=%s, post_process_flag=%d, summary_method_param=%s, SMC_time_limit=%d, model_type=%s, NUM_THREADS=%d"
           % (pop_size, M, alpha, NT, epsilon, cov_factor, cov_reduction_factor, noise_std, N, resampling_method, post_process_flag, summary_method_param, smc_time_limit, model_type, NUM_THREADS))
    print("\nTotal time needed to reach this result was",t_final-t_init)
    print("Total SMC time", t_final-t_SMC_init)
    print("\nResampled on the generations:")
    print(resampling_generation)
    print("Num perturbations per generation")
    print(num_simulations_per_generation)
    print("Time per generation")
    print(current_gen_time)
    print("Simulation Total times called",np.sum(num_simulations_per_generation))

    #Saving array:
    name = "N=" + str(N) + ",model=" + model_type + "," + str(np.datetime64('now')) +".npy"
    print("Saving array with name", name)
    np.save(name, current_particles)


    #Doing Plots:
    if(model_type == "linear" or model_type == "quadratic"):
        plot_gaussian_distributions(prior_means, prior_covs, posterior_means, posterior_covs, current_means, current_covs, current_corrected_means, current_corrected_covs, ground_truth, num_rows=int(num_params/2) , num_cols=2)
    
    if(model_type == "NN"):
        plot_non_gaussian_distributions(prior_means, prior_covs, MCMC_posterior, current_particles, current_corrected_samples, ground_truth, num_rows=int(num_params/2) , num_cols=2)


    plot_history(posterior_means, posterior_covs, populations_list, current_means, current_covs, corrected_means_list, corrected_covs_list, corrected_kl_divergence_list, type_flag = 1)
    if(post_process_flag ==1):
        plot_history(posterior_means, posterior_covs, populations_list, current_means, current_covs, corrected_means_list, corrected_covs_list, corrected_kl_divergence_list, type_flag = 2)
        plot_history(posterior_means, posterior_covs, populations_list, current_means, current_covs, corrected_means_list, corrected_covs_list, corrected_kl_divergence_list, type_flag = 3)
   
    #FINISHED PRESENTING THE RESULTS#


if __name__ == "__main__":
    main() 



