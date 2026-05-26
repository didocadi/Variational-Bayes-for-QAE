import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.stats import binom
from scipy.optimize import brute

def likelihood(a, ms, xs, N):
    theta = np.arcsin(np.sqrt(a))
    likelihood = []
    prob = 1
    for i in range(len(ms)):
        h_k = xs[i]
        prob *= (((np.sin((2*ms[i]+1)*theta))**2)**h_k)*(((np.cos((2*ms[i]+1)*theta))**2)**(N-h_k))
    return -prob
    
 #Define the brute-force search range
search_range = (slice(0.01, 0.99, 0.001),)


def optimize_likelihood(a,ms,xs,N):
    ms_cumulative = []
    xs_cumulative = []
    probs = []
    for m,x in zip(ms,xs):
        ms_cumulative.append(m)
        xs_cumulative.append(x)
        #Perform brute-force minimization
        mle_result = brute(likelihood,search_range,args=(ms_cumulative,xs_cumulative, N),full_output=True)
        probs.append(float(np.ravel(mle_result[0])[0]))
    return probs


def queries_inc(ms):
    sum = 0
    queries = []
    for i in range(len(ms)):
        sum += (1+2*(i+1))*100
        queries.append(sum)
    return queries


# Perform binomial sampling on the analytical probability
def binomial_sampling(m, a, N):
    #probabilities_analytical = []
    theta = np.arcsin(np.sqrt(a))

    # Get the theoretical probability from the formula
    prob_analytical = np.sin((2 * m + 1) * theta) ** 2
    # probabilities_analytical.append(prob_analytical)

    # Apply binomial sampling (simulate 100 shots)
    binomial_sample = binom.rvs(n=N, p=prob_analytical)

    return binomial_sample

def xs_calc(ms,a,N):
    xs = []
    for m in ms:
        xs.append(binomial_sampling(m,a,N))
    return xs


def rmse(Nruns,a,ms,N):
    all_probs = []
    for run in range(Nruns):
        xs = xs_calc(ms,a,N)
        probs = optimize_likelihood(a,ms,xs,N)
        all_probs.append(probs)
    all_probs = np.array(all_probs)
    # Compute RMSE at each step (across runs)
    rmse = np.sqrt(np.mean((all_probs - a) ** 2, axis=0))
    return rmse


if __name__ == "__main__":
    Nruns = 10
    a = 0.35
    ms = range(1,20)
    #ms = [1,2]
    #xs = [1,0]
    N = 100
    xs = xs_calc(ms,a,N)


    queries_cumulative = queries_inc(ms)
    errors = rmse(Nruns,a,ms,N)

    classical_error= 1 / np.sqrt(queries_cumulative) # Standard Quantum Limit
    quantum_error = 1 / np.array(queries_cumulative)

    print('probs = ', optimize_likelihood(a,ms,xs,N))

    mle_estimates = optimize_likelihood(a, ms, xs, N)

    plt.figure()
    plt.plot(ms, mle_estimates, label='Incremental MLE')
    plt.axhline(a, color='red', linestyle='--', label='True amplitude')

    plt.xlabel('Maximum Grover iteration included')
    plt.ylabel('Estimated amplitude')
    plt.title('Incremental Maximum Likelihood Estimates')
    plt.grid(True)
    plt.legend()
    plt.show()

    #print('binomial: = ', binomial_sampling(ms[8],a,N))

    #print('xs = ', xs_calc(ms,a,N))

    #print('MLE = ', MLE(a,ms,xs,N))

    #probabilities_binomial = binomial_sampling(ms, a,N)
    #plt.plot(ms, probabilities_binomial)
    #print('cumulative = ', queries_inc(ms))
    #print('MLE = ', MLE(a,ms,xs))
    #plt.plot(ms, likelihood(a,ms,xs,N))
    #plt.figure()
    #plt.scatter(queries_cumulative, errors)
    #plt.xlabel("Cumulative number of queries")
    #plt.ylabel("RMSE of amplitude estimates")
    #plt.title("Amplitude estimation RMSE vs Queries (Nruns=10)")
    #plt.grid(True)
    #plt.plot(ms, optimize_likelihood(a,ms,xs,N))
    plt.figure()
    plt.scatter(queries_cumulative[1:], errors[1:], label = "RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.loglog(queries_cumulative[1:], classical_error[1:]*0.3, label="Classical (SQL)", ls='--')
    plt.loglog(queries_cumulative[1:], quantum_error[1:]*4, label="Quantum (QAE, Heisenberg)", ls='--')
    plt.xlabel("Cumulative queries")
    plt.ylabel("Error")
    plt.title("Errors vs Queries (log–log scale, Nruns=10)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    #plt.plot(ms, MLE(a,ms,xs,N))
    plt.show()