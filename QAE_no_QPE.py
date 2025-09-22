import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.stats import binom

def likelihood(a, ms, xs, N):
    theta = np.arcsin(np.sqrt(a))
    likelihood = []
    for i in range(len(ms)):
        h_k = xs[i]
        prob = (((np.sin((2*ms[i]+1)*theta))**2)**h_k)*(((np.cos((2*ms[i]+1)*theta))**2)**(N-h_k))
        likelihood.append(prob)
    return likelihood

def MLE (a,ms, xs,N):
    mle = []
    mle_inc = 1
    ms_inc = []
    for m in ms:
        ms_inc.append(m)
        mle_inc *= max(likelihood(a,ms_inc,xs,N))
        mle.append(mle_inc)
    return mle

def queries_inc(ms):
    sum = 0
    queries = []
    for i in range(len(ms)):
        sum += (1+2*(i+1))*100
        queries.append(sum)
    return queries


# Perform binomial sampling on the analytical probability
def binomial_sampling(m_values, a, N):
    #probabilities_analytical = []
    probabilities_binomial = []
    theta = np.arcsin(np.sqrt(a))

    for m in m_values:
        # Get the theoretical probability from the formula
        prob_analytical = np.sin((2 * m + 1) * theta) ** 2
       # probabilities_analytical.append(prob_analytical)

        # Apply binomial sampling (simulate 100 shots)
        binomial_sample = binom.rvs(n=N, p=prob_analytical) / N
        probabilities_binomial.append(binomial_sample)

    return probabilities_binomial

def xs_calc(ms,a,N):
    xs = []
    probabilities_binomial = binomial_sampling(ms,a,N)
    for i in range(len(ms)):
        if probabilities_binomial[i] >= 0.9:
            xs.append(1)
        else: xs.append(0)
    return xs


a = 0.35
ms = range(1,31)
#ms = [1,2]
#xs = [1,0]
N = 100
xs = xs_calc(ms,a,N)

#print('xs = ', xs_calc(ms,a,N))

print('MLE = ', MLE(a,ms,xs,N))

probabilities_binomial = binomial_sampling(ms, a,N)
#plt.plot(ms, probabilities_binomial)
#print('cumulative = ', queries_inc(ms))
#print('MLE = ', MLE(a,ms,xs))
#plt.plot(ms, likelihood(a,ms,xs,N))
plt.plot(ms, MLE(a,ms,xs,N))
plt.show()