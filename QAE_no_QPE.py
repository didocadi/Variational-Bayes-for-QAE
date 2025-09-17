import numpy as np
import math
import matplotlib.pyplot as plt
import random

def likelihood(a, ms, xs, N):
    theta = np.arcsin(np.sqrt(a))
    h_k = len(xs)
    likelihood = []
    for i in range(len(ms)):
        likelihood_m = (((np.sin((2*ms[i]+1)*theta))**2)**h_k)*(((np.cos((2*ms[i]+1)*theta))**2)**(N-h_k))
        likelihood.append(likelihood_m)
    return likelihood

def MLE (a,ms, xs):
    queries = queries_inc(ms)
    mle = []
    for i in range(len(queries)):
        N = queries[i]
        mle_i = max(likelihood(a,ms,xs,N))
        mle.append(mle_i)
    return mle

def queries_inc(ms):
    sum = 0
    queries = []
    for i in range(len(ms)):
        sum += (1+2*(i+1))*100
        queries.append(sum)
    return queries

a = 0.35
ms = range(1,31)
xs = [1,2]
N = 100

#print('cumulative = ', queries_inc(ms))
#print('MLE = ', MLE(a,ms,xs))
plt.plot(ms, MLE(a,ms,xs))
plt.show()