import numpy as np
import math
import matplotlib.pyplot as plt
import random

from scipy.optimize import brute
from qiskit_aer import Aer
from qiskit import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT

from QAE import grover_operator, QAE_circuit


def likelihood (a,xs,t):
    prob = 1
    theta = np.arcsin(np.sqrt(a))
    M = 2**t
    for i in range(len(xs)):
        x = xs[i]
        delta_pos = abs((theta/np.pi) - (x/M)) % 1
        delta_neg = abs(1 - (theta/np.pi) - (x/M)) % 1
        prob_qpe_phi = ((np.sin(M*delta_pos*np.pi))**2)/((M**2)*(np.sin(delta_pos*np.pi))**2)
        prob_qpe_phi_neg = ((np.sin(M*delta_neg*np.pi))**2)/((M**2)*(np.sin(delta_neg*np.pi))**2)
        prob_qae = (prob_qpe_phi + prob_qpe_phi_neg)/2
        prob *= prob_qae

    return prob

#xs = [0,1,2,3,4,5,6,7]

#probs = likelihood(a,xs,t)

#print(f'probability of measuring x = {xs} with amplitude = {a} is: {probs}')


def likelihood_a (a_s,x,t):
    probs = []
    M = 2**t
    for i in range(len(a_s)):
        a = a_s[i]
        theta = np.arcsin(np.sqrt(a))
        delta_pos = abs((theta/np.pi) - (x/M))
        delta_neg = abs(1 - (theta/np.pi) - (x/M))
        prob_qpe_phi = ((np.sin(M*delta_pos*np.pi))**2)/((M**2)*(np.sin(delta_pos*np.pi))**2)
        prob_qpe_phi_neg = ((np.sin(M*delta_neg*np.pi))**2)/((M**2)*(np.sin(delta_neg*np.pi))**2)
        prob_qae = (prob_qpe_phi + prob_qpe_phi_neg)/2
        probs.append(prob_qae)

    return probs

def likelihood_alt (a_s,x_s,t):
    probs = []
    for i in range(len(a_s)):
        a = a_s[i]
        probs.append(likelihood(a,x_s,t))
    
    return probs

def neg_log_likelihood(a, xs, t):
    # a is a numpy array when called from brute, so extract scalar
    a_val = a[0] if isinstance(a, (list, np.ndarray)) else a
    # Check domain to avoid invalid arcsin
    if a_val <= 0 or a_val >= 1:
        return np.inf
    try:
        prob = likelihood(a_val, xs, t)
        if prob <= 0:
            return np.inf
        return -np.log(prob)
    except:
        return np.inf  # catch rare numerical issues

def classical_estimation(a, num_shots):
    successes = 0
    for _ in range(num_shots):
        if random.random() < a:
            successes += 1
    return successes / num_shots

# Metrology limits
def plot_metrology_limits():
    N = np.logspace(1, 4, 100) # number of queries from 10 to 10,000


    # Error scaling laws
    classical_error = 1 / np.sqrt(N) # Standard Quantum Limit
    quantum_error = 1 / N # Heisenberg Limit


    plt.figure(figsize=(7, 5))
    plt.loglog(N, classical_error, label="Classical (SQL)", linewidth=2)
    plt.loglog(N, quantum_error, label="Quantum (QAE, Heisenberg)", linewidth=2)


    plt.xlabel("Number of queries N")
    plt.ylabel("Estimation error Δa")
    plt.title("Fundamental Limits of Metrology")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.show()
    


if __name__ == "__main__":
    # Example Usage
    n = 3  # Number of qubits
    k = 2  # Number of good states
    t = 4  # Number of counting qubits
    real_amplitude = k/(2**n)

    print("real amplitude: ", real_amplitude)

    U = QuantumCircuit(n)
    U.h(range(n))

    QAE = QAE_circuit(n,k,t,U)

    QAE.measure(range(t), range(t))
    QAE.draw(fold=-1)



    backend = Aer.get_backend('qasm_simulator')
    results = {}

    compiled_circuit = transpile(QAE, backend)
    shots = 100
    result = backend.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    xs_final = []
    for key in counts.keys():
        xs_final.extend(int(key,2) for _ in range(counts[key]))


   # print('xs_final:', xs_final)

    a_s = np.linspace(0,1,500)

    probs_a = likelihood_alt(a_s,xs_final,t)
    #print('probs_a: ', probs_a)

    #Define the brute-force search range
    search_range = (slice(0.01, 0.99, 0.001),)

    #Perform brute-force minimization
    mle_result = brute(neg_log_likelihood,search_range,args=(xs_final, t),full_output=True)

    # Extract MLE result
    mle_amplitude = mle_result[0]
    max_log_likelihood = -mle_result[1]
    y_mle = likelihood(mle_amplitude, xs_final, t)

    print('MLE_amplitude:', mle_amplitude)

    measured_str = max(counts, key=counts.get)

    measured_int = int(measured_str, 2)

    amplitude = np.sin(np.pi*measured_int/(2**t))**2

    print('QAE amplitude:', amplitude)

    plt.plot(a_s, probs_a, label='Likelihood')
    plt.plot(mle_amplitude, y_mle, 'r*', markersize=12, label='MLE')
    plt.axvline(real_amplitude, color='red', linestyle='--', label='Real Amplitude')
    plt.axvline(amplitude, color='black', linestyle='--', label='QAE Amplitude')
    plt.xlabel('Amplitude')
    plt.ylabel('Probability')
    #plt.title('Likelihood vs Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    true_a = random.random() # pick random probability in [0,1]
    estimate = classical_estimation(true_a, 10000)
    print(f"True a = {true_a:.4f}, Classical estimate = {estimate:.4f}")

    plot_metrology_limits()