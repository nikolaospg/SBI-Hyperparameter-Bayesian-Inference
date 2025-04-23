import numpy as np
from tabulate import tabulate 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 



def plot_gaussian_distributions(prior_mean, prior_cov,
                                 posterior_mean, posterior_cov,
                                 est_posterior_mean, est_posterior_cov,
                                 corrected_mean, corrected_cov,
                                 ground_truth,
                                 num_rows, num_cols):

    def format_array(arr):
        if arr.ndim == 1:
            return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"
        return np.array2string(arr, precision=6, suppress_small=True)

    # Build table data
    table_data = [
        ["Prior", format_array(prior_mean), format_array(prior_cov)],
        ["Posterior", format_array(posterior_mean), format_array(posterior_cov)],
        ["ABC Posterior", format_array(est_posterior_mean), format_array(est_posterior_cov)]
    ]
    if corrected_mean is not None and corrected_cov is not None:
        table_data.append(["Corrected ABC Posterior", format_array(corrected_mean), format_array(corrected_cov)])

    print("*** Final Results ***")
    print("Ground truth params:", ground_truth)
    print(tabulate(table_data, headers=["Distribution", "Mean", "Covariance"], tablefmt="pretty"))

    def plot_marginals(include_prior, title_suffix, distributions, colors, styles):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 3 * num_rows))
        axes = axes.flatten()
        num_vars = len(prior_mean)

        for i in range(num_vars):
            for (label, mean, cov), color, style in zip(distributions, colors, styles):
                std = np.sqrt(cov[i, i])
                x = np.linspace(mean[i] - 3 * std, mean[i] + 3 * std, 1000)
                pdf = multivariate_normal(mean[i], cov[i, i]).pdf(x)
                axes[i].plot(x, pdf, label=label, linestyle=style, color=color, lw=1.5)
                axes[i].plot(mean[i], pdf[np.argmax(pdf)], 'x', color=color, markersize=8)

            axes[i].set_title(f"Θ[{i}]")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")
            axes[i].legend()

        for j in range(num_vars, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title_suffix)
        plt.tight_layout(pad=2)

    # Distributions and style setup
    base_distributions = [
        ("Posterior", posterior_mean, posterior_cov),
        ("ABC Posterior", est_posterior_mean, est_posterior_cov),
    ]
    base_colors = ["#ff7f0e", "#2ca02c"]
    base_styles = ["-", "-"]

    if corrected_mean is not None and corrected_cov is not None:
        base_distributions.append(("Corrected ABC Posterior", corrected_mean, corrected_cov))
        base_colors.append("#d62728")
        base_styles.append("-")

    # With prior
    plot_marginals(
        include_prior=True,
        title_suffix="Distributions With Prior",
        distributions=[("Prior", prior_mean, prior_cov)] + base_distributions,
        colors=["#1f77b4"] + base_colors,
        styles=["-."] + base_styles
    )

    # Without prior
    plot_marginals(
        include_prior=False,
        title_suffix="Distributions Without Prior",
        distributions=base_distributions,
        colors=base_colors,
        styles=base_styles
    )

    plt.show()


def plot_non_gaussian_distributions(prior_means, prior_covs,
                                    MCMC_posterior, ABC_empirical_samples,
                                    corrected_samples, ground_truth,
                                    num_rows, num_cols):

    def format_array(arr):
        if arr.ndim == 1:
            return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"
        return np.array2string(arr, precision=6, suppress_small=True)

    # Table data
    table_data = [
        ["Prior", format_array(prior_means), format_array(prior_covs)],
        ["MCMC Posterior", format_array(np.mean(MCMC_posterior, axis=0)), format_array(np.cov(MCMC_posterior.T))],
        ["ABC Posterior", format_array(np.mean(ABC_empirical_samples, axis=0)), format_array(np.cov(ABC_empirical_samples.T))]
    ]
    if corrected_samples is not None:
        table_data.append(["Corrected ABC Posterior", format_array(np.mean(corrected_samples, axis=0)), format_array(np.cov(corrected_samples.T))])

    print("*** Final Results ***")
    print("Ground truth params:", ground_truth)
    print(tabulate(table_data, headers=["Distribution", "Mean", "Covariance"], tablefmt="pretty"))

    num_vars = prior_means.shape[0]

    def plot_histograms(title_suffix, datasets, labels, colors, include_prior=True, bin_flag=0):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 3 * num_rows))
        axes = axes.flatten()

        for i in range(num_vars):
            if include_prior:
                std = np.sqrt(prior_covs[i, i])
                x = np.linspace(prior_means[i] - 3 * std, prior_means[i] + 3 * std, 1000)
                pdf = multivariate_normal(prior_means[i], prior_covs[i, i]).pdf(x)
                axes[i].plot(x, pdf, linestyle='--', color='black', label='Prior', lw=1.5)

            lengths =[]
            for current_dataset in datasets:
                current_length = len(current_dataset)
                lengths.append(current_length)
            min_length = min(lengths)

            for data, label, color in zip(datasets, labels, colors):
                N = len(data)
                if(bin_flag == 0):
                    n_bins=int(np.sqrt(N))
                    extra_title = ""
                elif(bin_flag == 1):
                    n_bins=int(np.sqrt(min_length))
                    extra_title = ", Equal Num Bins"
                    
                axes[i].hist(data[:, i], bins=n_bins, density=True, alpha=0.6,
                             label=label, edgecolor='black', linewidth=0.5, color=color)

            axes[i].set_title(f"Θ[{i}]")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")
            axes[i].legend()

        for j in range(num_vars, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title_suffix  + extra_title)
        plt.tight_layout(pad=2)

    # Setup base data
    datasets = [MCMC_posterior, ABC_empirical_samples]
    labels = ["MCMC Posterior", "ABC Posterior"]
    colors = ["#ff7f0e", "#2ca02c"]

    if corrected_samples is not None:
        datasets.append(corrected_samples)
        labels.append("Corrected ABC Posterior")
        colors.append("#d62728")

    # Plot 1: with prior
    plot_histograms(
        title_suffix="Distributions With Prior",
        datasets=datasets,
        labels=labels,
        colors=colors,
        include_prior=True
    )

    # Plot 2: without prior
    plot_histograms(
        title_suffix="Distributions Without Prior",
        datasets=datasets,
        labels=labels,
        colors=colors,
        include_prior=False
    )

    # Plot 3: MCMC vs ABC only
    plot_histograms(
        title_suffix="Distributions, No Correction",
        datasets=[MCMC_posterior, ABC_empirical_samples],
        labels=["MCMC Posterior", "ABC Posterior"],
        colors=["#ff7f0e", "#2ca02c"],
        include_prior=False
    )

    # Plot 3: MCMC vs ABC only
    plot_histograms(
        title_suffix="Distributions, No Correction",
        datasets=[MCMC_posterior, ABC_empirical_samples],
        labels=["MCMC Posterior", "ABC Posterior"],
        colors=["#ff7f0e", "#2ca02c"],
        include_prior=False,
        bin_flag= 1
    )

    plt.show()



def plot_history(posterior_means, posterior_covs, populations_list, final_mean, final_cov, corrected_means, corrected_covs, corrected_kl_divergences, type_flag="1"):
    'If the type_flag ==1 then we plot uncorrected, if the type_flag==2 then we plot corrected history, if type_flag==3 the we plot both'

    def plot_function_sole(posterior_means, posterior_covs, mean_history, covariance_history, kl_divergence_history, type_flag):

        if(type_flag == 1):
            title = "Parameter Estimation History, Uncorrected"
            title_kl = "KL Divergence History, Uncorrected"
        if(type_flag == 2):
            title = "Parameter Estimation History, Corrected"
            title_kl = "KL Divergence History, Corrected"
        if(type_flag == 3):
            print("plot_function sole should not be called for depicting both corrected and uncorrected")
            return 
        
        mean_history = np.array(mean_history)  # Shape: (num_generations, 2)
        covariance_history = np.array(covariance_history)  # Shape: (num_generations, 2, 2)

        # Number of generations
        num_generations = len(mean_history)

        # Create a figure with 3 rows and 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        fig.suptitle(title, fontsize=16)

        # Plot the mean history
        for i in range(2):  # Loop over the two parameters
            axes[0, i].plot(range(num_generations), mean_history[:, i], label=rf"Estimated mean $\theta_{i+1}$")
            axes[0, i].axhline(posterior_means[i], color='r', linestyle='--', label=rf"Theoretical mean $\theta_{i+1}$")
            axes[0, i].set_xlabel("Generation")
            axes[0, i].set_ylabel(rf"$\theta_{i+1}$")
            axes[0, i].legend()
            axes[0, i].grid()

        # Plot the first row of the covariance matrix history
        for i in range(2):  # Loop over the two elements in the first row
            axes[1, i].plot(range(num_generations), covariance_history[:, 0, i], label=f"Estimated Cov[$\\theta_1, \\theta_{i+1}$]")
            axes[1, i].axhline(posterior_covs[0, i], color='r', linestyle='--', label=f"Theoretical Cov[$\\theta_1, \\theta_{i+1}$]")
            axes[1, i].set_xlabel("Generation")
            axes[1, i].set_ylabel(f"Cov[$\\theta_1, \\theta_{i+1}$]")
            axes[1, i].legend()
            axes[1, i].grid()

        # Plot the second row of the covariance matrix history
        for i in range(2):  # Loop over the two elements in the second row
            axes[2, i].plot(range(num_generations), covariance_history[:, 1, i], label=f"Estimated Cov[$\\theta_2, \\theta_{i+1}$]")
            axes[2, i].axhline(posterior_covs[1, i], color='r', linestyle='--', label=f"Theoretical Cov[$\\theta_2, \\theta_{i+1}$]")
            axes[2, i].set_xlabel("Generation")
            axes[2, i].set_ylabel(f"Cov[$\\theta_2, \\theta_{i+1}$]")
            axes[2, i].legend()
            axes[2, i].grid()

        # Adjust layout 
        plt.tight_layout()

        # Creating figure for KL divergence 
        plt.figure()
        plt.plot(range(num_generations), kl_divergence_history, label=rf"KL Divergence")
        plt.xlabel("Generation")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid()
        plt.title(title_kl)
        plt.show()

    def plot_function_both(posterior_means, posterior_covs, mean_history, covariance_history, corrected_means, corrected_covs, kl_divergence_history, corrected_kl_divergences, type_flag):

        if(type_flag != 3):
            print("plot_function_both should be called only to depict both the corrected and uncorrected")
            return 
        
        # Convert lists to numpy arrays for easier indexing
        mean_history = np.array(mean_history)  # Shape: (num_generations, 2)
        covariance_history = np.array(covariance_history)  # Shape: (num_generations, 2, 2)
        corrected_means = np.array(corrected_means)  # Shape: (num_generations, 2)
        corrected_covs = np.array(corrected_covs)  # Shape: (num_generations, 2, 2)

        # Number of generations
        num_generations = len(mean_history)

        # Create a figure with 3 rows and 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle("Parameter Estimation History", fontsize=16)

        # Plot the mean history
        for i in range(2):  # Loop over the two parameters
            axes[0, i].plot(range(num_generations), mean_history[:, i], label=rf"Estimated mean $\theta_{i+1}$")
            axes[0, i].plot(range(num_generations), corrected_means[:, i], label=rf"Corrected mean $\theta_{i+1}$", color='green', linestyle='-')
            axes[0, i].axhline(posterior_means[i], color='r', linestyle='--', label=rf"Theoretical mean $\theta_{i+1}$")
            axes[0, i].set_xlabel("Generation")
            axes[0, i].set_ylabel(rf"$\theta_{i+1}$")
            axes[0, i].legend()
            axes[0, i].grid()

        # Plot the first row of the covariance matrix history
        for i in range(2):  # Loop over the two elements in the first row
            axes[1, i].plot(range(num_generations), covariance_history[:, 0, i], label=f"Estimated Cov[$\\theta_1, \\theta_{i+1}$]")
            axes[1, i].plot(range(num_generations), corrected_covs[:, 0, i], label=f"Corrected Cov[$\\theta_1, \\theta_{i+1}$]", color='green', linestyle='-')
            axes[1, i].axhline(posterior_covs[0, i], color='r', linestyle='--', label=f"Theoretical Cov[$\\theta_1, \\theta_{i+1}$]")
            axes[1, i].set_xlabel("Generation")
            axes[1, i].set_ylabel(f"Cov[$\\theta_1, \\theta_{i+1}$]")
            axes[1, i].legend()
            axes[1, i].grid()

        # Plot the second row of the covariance matrix history
        for i in range(2):  # Loop over the two elements in the second row
            axes[2, i].plot(range(num_generations), covariance_history[:, 1, i], label=f"Estimated Cov[$\\theta_2, \\theta_{i+1}$]")
            axes[2, i].plot(range(num_generations), corrected_covs[:, 1, i], label=f"Corrected Cov[$\\theta_2, \\theta_{i+1}$]", color='green', linestyle='-')
            axes[2, i].axhline(posterior_covs[1, i], color='r', linestyle='--', label=f"Theoretical Cov[$\\theta_2, \\theta_{i+1}$]")
            axes[2, i].set_xlabel("Generation")
            axes[2, i].set_ylabel(f"Cov[$\\theta_2, \\theta_{i+1}$]")
            axes[2, i].legend()
            axes[2, i].grid()

        # Adjust layout and show the plot
        plt.tight_layout()
        
        plt.figure()
        plt.plot(range(num_generations), kl_divergence_history, label=rf"Uncorrected KL Divergence$")
        plt.plot(range(num_generations), corrected_kl_divergences, color="green", label=rf"Corrected KL Divergence$")
        plt.xlabel("Generation")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid()
        plt.title("KL Divergence History, both")
        plt.show()
    
    #First crafting the lists- np arrays that will hold the desired values
    num_pops = len(populations_list)
    mean_history = []
    covariance_history = [ ]
    kl_divergence_history =[]
    for gen_index in range(num_pops):
        current_pop = populations_list[gen_index]
        current_mean = current_pop["means"]
        current_cov = current_pop["covs"]   
        current_kl_divergence = current_pop["kl_divergence"]
        mean_history.append(current_mean)
        covariance_history.append(current_cov)
        kl_divergence_history.append(current_kl_divergence)
    mean_history.append(final_mean)
    covariance_history.append(final_cov)
    kl_divergence_history.append(current_kl_divergence)
    
    #Finished crafting the history list 
    if(type_flag!=1):
        corrected_means = np.concatenate(([corrected_means[0]], corrected_means))
        corrected_covs = np.concatenate(([corrected_covs[0]], corrected_covs))
        corrected_kl_divergences = np.concatenate(([corrected_kl_divergences[0]], corrected_kl_divergences))

    if(type_flag == 1):
        plot_function_sole(posterior_means, posterior_covs, mean_history, covariance_history, kl_divergence_history, type_flag)
    if(type_flag == 2):
        plot_function_sole(posterior_means, posterior_covs, corrected_means, corrected_covs, corrected_kl_divergences, type_flag)
    if(type_flag == 3):
        plot_function_both(posterior_means, posterior_covs, mean_history, covariance_history, corrected_means, corrected_covs, kl_divergence_history, corrected_kl_divergences, type_flag)


