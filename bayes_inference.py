import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

m_s = [1,2,3]
a_s = np.arange(0.01,1,0.01)
n = 10
k_s = [5,7,10]


def posterior_prob(m_s, k_s, n, a_s):
    """Compute posterior probability P(a | data up to given m,k)."""
    likelihood = np.ones_like(a_s)
    for m, k in zip(m_s, k_s):
        theta = np.arcsin(np.sqrt(a_s))
        p_good = np.sin((2 * m + 1) * theta) ** 2 
        likelihood *= binom.pmf(k, n, p_good)
    posterior = likelihood / np.sum(likelihood)  # normalization
    return posterior

post1 = posterior_prob(m_s[:1], k_s[:1], n, a_s)
post2 = posterior_prob(m_s[:2], k_s[:2], n, a_s)
post3 = posterior_prob(m_s[:3], k_s[:3], n, a_s)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(a_s, post1, label='m=1, k=5')
plt.plot(a_s, post2, label='m=[1,2], k=[5,7]')
plt.plot(a_s, post3, label='[m=1,2,3], k=[5,7,10]')
plt.xlabel("Amplitude a")
plt.ylabel("Posterior Probability Density")
plt.title("Bayesian Posterior for Amplitude a (Cumulative Updates)")
plt.legend()
plt.grid(True)
plt.show()

def likelihoods(m, a_s, n, k):
    likelihoods = []
    better_a = a_s[0]
    for a in a_s:
        theta = np.arcsin(np.sqrt(a))
        p_good = np.sin((2 * m + 1) * theta) ** 2
        likelihood = binom.pmf(k, n, p_good)
        likelihoods.append(likelihood)
    better_a = a_s[likelihoods.index(max(likelihoods))]
    return better_a, likelihoods

print("Better estimate of amplitude = ", likelihoods(1,[0.1,0.75],n,8)[0])
#print(f'{likelihoods(m,a_s,n,k)[0]} is {likelihoods(m,a_s,n,k)[1][0]/likelihoods(m,a_s,n,k)[1][1]} times greater than {a_s[1]}')
