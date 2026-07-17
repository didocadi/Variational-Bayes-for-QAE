import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from bayes_gaussian import queries_inc

def generate_data(m_s, a_true, n):
    k_s = []
    theta_true = np.arcsin(np.sqrt(a_true))

    for m in m_s:
        p_good = np.sin((2 * m + 1) * theta_true) ** 2
        k = np.random.binomial(n, p_good)
        k_s.append(k)

    return k_s


def posterior_prob(m_s, k_s, n, a_s):
    """Compute posterior probability P(a | data up to given m,k)."""
    likelihood = np.ones_like(a_s)
    posteriors = []
    for m, k in zip(m_s, k_s):
        theta = np.arcsin(np.sqrt(a_s))
        p_good = np.sin((2 * m + 1) * theta) ** 2 
        likelihood *= binom.pmf(k, n, p_good)
        posterior = likelihood / np.sum(likelihood)  # normalization
        posteriors.append(posterior)
    return posteriors

def posterior_gaussian(m_s, k_s, n, nsamples=20000):

    a_s = np.random.rand(nsamples)
    posteriors = []

    for m, k in zip(m_s, k_s):

        thetas = np.arcsin(np.sqrt(a_s))
        p_good = np.sin((2 * m + 1) * thetas) ** 2

        likelihoods = binom.pmf(k, n, p_good)

        norms = likelihoods / np.sum(likelihoods)

        mean = np.sum(norms * a_s)
        var = np.sum(norms * (a_s - mean) ** 2)
        std = np.sqrt(var)

        posteriors.append((mean, std))

        # Gaussian update step
        a_s = np.random.normal(mean, std, nsamples)
        a_s = np.clip(a_s, 0.0, 1.0)

    return posteriors

#def posterior_mean_error(post, a_true):

#    means = np.array([m for m, s in post])
#    errors = np.abs(means - a_true)

 #   return means, errors


#means, errors = posterior_mean_error(post, a_true)

#print("\nPosterior means:", means)
#print("Absolute errors:", errors)

def rmse_exact(Nruns, m_s, a_true, n, a_grid):

    all_means = []

    for _ in range(Nruns):

        # Generate new experimental data
        k_s = generate_data(m_s, a_true, n)

        # Sequential exact posteriors
        posteriors = posterior_prob(m_s, k_s, n, a_grid)

        # Posterior mean after each experiment
        means = [np.sum(a_grid * posterior) for posterior in posteriors]

        all_means.append(means)

    all_means = np.array(all_means)

    rmse = np.sqrt(np.mean((all_means - a_true)**2, axis=0))

    return rmse

def rmse_gaussian(Nruns, m_s, a_true, n, nsamples=20000):

    all_means = []

    for _ in range(Nruns):

        # Generate new experimental data
        k_s = generate_data(m_s, a_true, n)

        # Sequential Gaussian approximations
        posteriors = posterior_gaussian(m_s, k_s, n, nsamples)

        means = [mean for mean, std in posteriors]

        all_means.append(means)

    all_means = np.array(all_means)

    rmse = np.sqrt(np.mean((all_means - a_true)**2, axis=0))

    return rmse


if __name__ == "__main__":

    a_true = 0.95
    m_s = np.random.randint(1,20,5)
    #np.random.randint(1,20,4)
    m_s.sort()
    print(m_s)
    n = 1
    a_grid = np.linspace(0, 1, 1000)

    k_s = generate_data(m_s, a_true, n)
    print("Generated k_s:", k_s)

    exact_posts = posterior_prob(m_s, k_s, n, a_grid)
    gaussian_posts = posterior_gaussian(m_s, k_s, n)

    fig, ax = plt.subplots(figsize=(10,6))

    # reconstruct Gaussian curves
    for i, (mean, std) in enumerate(gaussian_posts, start=1):
        gaussian = np.exp(-0.5 * ((a_grid - mean) / (std + 1e-8))**2)
        gaussian /= np.sum(gaussian)

        exact = exact_posts[i-1]

        if (i < 10):

            line = ax.plot(a_grid[(a_grid > 0) & (a_grid < 1)], exact[(a_grid > 0) & (a_grid < 1)], lw=2, label=f"Reference: m=[{min(m_s)}...{m_s[i-1]}]", alpha=0.4)[0]
            ax.plot(a_grid[(a_grid > 0) & (a_grid < 1)], gaussian[(a_grid > 0) & (a_grid < 1)], '--', lw=2, color=line.get_color(),
                label=f"Gaussian: m=[{min(m_s)}...{m_s[i-1]}]")

    plt.axvline(a_true, color='black', linestyle='--', label=f"true amplitude (a = {a_true})", alpha = 0.7)

    ax.set_xlabel("Amplitude $a$")
    ax.set_ylabel("Posterior Probability")
    ax.set_title("Case 1")
    ax.grid(True)
    ax.legend()
    plt.show()

    Nruns = 10

    rmse_exact_values = rmse_exact(Nruns, m_s, a_true, n, a_grid)
    rmse_gaussian_values = rmse_gaussian(Nruns, m_s, a_true, n)
    queries_cumulative = queries_inc(m_s)

    #plt.figure(figsize=(7,5))
    #plt.plot(m_s, rmse_exact_values, 'o-', label="Reference posteriors")
    #plt.plot(m_s, rmse_gaussian_values, 's--', label="Gaussian approximation")

    #plt.xlabel("Maximum Grover iteration")
    #plt.ylabel("RMSE")
    #plt.grid(True)
    #plt.legend()
    #plt.title("Case 8")
    #plt.show()


    classical_error = 1 / np.sqrt(queries_cumulative) # Standard Quantum Limit
    quantum_error = 1 / np.array(queries_cumulative)

    plt.figure()
    plt.scatter(queries_cumulative, rmse_gaussian_values, label = "Gaussian RMSE")
    plt.scatter(queries_cumulative, rmse_exact_values, label = "Posteriors RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.loglog(queries_cumulative, classical_error, label="Classical (SQL)", ls='--')
    plt.loglog(queries_cumulative, quantum_error, label="Quantum (QAE, Heisenberg)", ls='--')
    plt.xlabel("Cumulative queries")
    plt.ylabel("Error")
    plt.title("Case 10 Errors vs Queries (log–log scale, Nruns=10)")
    plt.grid(True, which="both", ls="--")
    plt.legend(loc='lower left')
    #plt.plot(ms, MLE(a,ms,xs,N))
    plt.show()