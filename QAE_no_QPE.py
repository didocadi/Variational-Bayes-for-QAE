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
    return prob

def neg_log_likelihood(a,ms,xs,N):
    # a is a numpy array when called from brute, so extract scalar
    a_val = a[0] if isinstance(a, (list, np.ndarray)) else a
    # Check domain to avoid invalid arcsin
    if a_val <= 0 or a_val >= 1:
        return np.inf
    try:
        prob = likelihood(a_val,ms,xs,N)
        if prob <= 0:
            return np.inf
        return -np.log(prob)
    except:
        return np.inf  # catch rare numerical issues
    
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
        mle_result = brute(neg_log_likelihood,search_range,args=(ms_cumulative,xs_cumulative, N),full_output=True)
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


a = 0.35
ms = range(1,31)
#ms = [1,2]
#xs = [1,0]
N = 100
xs = xs_calc(ms,a,N)

print('probs = ', optimize_likelihood(a,ms,xs,N))

#print('binomial: = ', binomial_sampling(ms[8],a,N))

#print('xs = ', xs_calc(ms,a,N))

#print('MLE = ', MLE(a,ms,xs,N))

#probabilities_binomial = binomial_sampling(ms, a,N)
#plt.plot(ms, probabilities_binomial)
#print('cumulative = ', queries_inc(ms))
#print('MLE = ', MLE(a,ms,xs))
#plt.plot(ms, likelihood(a,ms,xs,N))
plt.plot(ms, optimize_likelihood(a,ms,xs,N))
#plt.plot(ms, MLE(a,ms,xs,N))
plt.show()