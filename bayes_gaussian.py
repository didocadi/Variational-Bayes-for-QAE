import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from QAE_no_QPE import binomial_sampling, xs_calc


def posterior_gaussian(m_s, k_s, n, nsamples = 20000):
    a_s = np.random.rand(nsamples)
    posteriors = []
    for m, k in zip(m_s, k_s):
        thetas = np.arcsin(np.sqrt(a_s))
        p_good = np.sin((2 * m + 1) * thetas) ** 2 
        likelihoods = (p_good ** k) * ((1 - p_good) ** (n - k))
        norms = likelihoods / np.sum(likelihoods)  # normalization
        mean = np.sum(norms * a_s)
        var = np.sum(norms * (a_s - mean) ** 2)
        std = np.sqrt(var)
        posteriors.append((mean, std))
        # Update samples from Gaussian posterior (next prior)
        a_s = np.random.normal(mean, std, nsamples)
        a_s = np.clip(a_s, 0.0, 1.0)
    return posteriors

def queries_inc(ms):
    sum = 0
    queries = []
    for i in range(len(ms)):
        sum += (1+2*(i+1))*100
        queries.append(sum)
    return queries

def rmse(Nruns, a_true, ms, Nshots, nsamples=20000):
    all_means = []
    for run in range(Nruns):
        xs = xs_calc(ms, a_true, Nshots)  # same as before (binomial sampling)

        posteriors = posterior_gaussian(ms, xs, Nshots, nsamples)

        means = np.array([mean for mean, std in posteriors])
        all_means.append(means)

    all_means = np.array(all_means)
    rmse = np.sqrt(np.mean((all_means - a_true) ** 2, axis=0))

    return rmse

if __name__ == "__main__":
    m_s = [1,2,3]
    n = 10
    k_s = [5,7,10]

    posteriors = posterior_gaussian(m_s, k_s, n)

    print("8(a) Sequential Gaussian posterior approximations:")
    for i, (mean, std) in enumerate(posteriors, start=1):
        print(f" After dataset {i} (m={m_s[i-1]}): mean = {mean:.4f}, std = {std:.4f}")    

    Nruns = 10
    a_true = 0.35
    ms = range(1,20)
    Nshots = 100

    errors = rmse(Nruns, a_true, ms, Nshots)
    queries_cumulative = queries_inc(ms)

    classical_error= 1 / np.sqrt(queries_cumulative) # Standard Quantum Limit
    quantum_error = 1 / np.array(queries_cumulative)

    plt.figure()
    plt.scatter(queries_cumulative[1:], errors[1:], label = "RMSE_gaussian")
    plt.xscale("log")
    plt.yscale("log")
    plt.loglog(queries_cumulative[1:], classical_error[1:], label="Classical (SQL)", ls='--')
    plt.loglog(queries_cumulative[1:], quantum_error[1:], label="Quantum (QAE, Heisenberg)", ls='--')
    plt.xlabel("Cumulative queries")
    plt.ylabel("Error")
    plt.title("Errors vs Queries (log–log scale, Nruns=10)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    #plt.plot(ms, MLE(a,ms,xs,N))
    plt.show()

