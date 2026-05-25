#initialization
import qiskit
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit_aer import Aer
from qiskit import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Operator

# import basic plot tools
from qiskit.visualization import plot_histogram

def create_grover_circuit(m, n, k, U):
    N = 2**n
    assert k < N, "k must be less than N"
    #calculate ideal number of iterations m
    #m = int(np.pi/4 * np.sqrt(N/k)-1/2)
    
    # Choose k good states randomly
    #good_states = np.random.choice(N, k, replace=False)
    good_states = [4,7]
    print(f'good states: {good_states}')
    
    # Construct the oracle that marks good states
    oracle = QuantumCircuit(n)
    for state in good_states:
        binary_state = format(state, f'0{n}b')
        
        # Apply X gates to flip the required qubits
        for qubit, bit in enumerate(binary_state):
            if bit == '0':
                oracle.x(qubit)
                
        # Apply multi-controlled Z (MCX + H)
        # oracle.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)
        
        # Undo X gates
        for qubit, bit in enumerate(binary_state):
            if bit == '0':
                oracle.x(qubit)
    
    # Construct the diffusion operator
    diffuser = QuantumCircuit(n)
    # Start with the conjugate transpose of the state preparation (U†)
    diffuser.append(U.inverse(), range(n))
    # Apply the reflection around |0⟩
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    # Reapply the state preparation (U)
    diffuser.append(U, range(n))
    
    # Combine the Grover operator
    grover_op = oracle.compose(diffuser)
    G = Operator(grover_op)
    
    grover_circuit = QuantumCircuit(n)
    # Apply the initialization operator
    grover_circuit.append(U, range(n))  # General initialization
    for _ in range(m):
        grover_circuit.append(G, range(n))

    return grover_circuit

# Example Usage
n = 3  # Number of qubits/2^n states total
k = 2  # Number of good states
#m = 3

U = QuantumCircuit(n)
U.h(range(n))

# Simulation for multiple iterations
backend = Aer.get_backend('qasm_simulator')
#results = {}

#grover_circuit = create_grover_circuit(m,n,k,U)

#grover_circuit.draw()

#grover_circuit.measure_all()
#compiled_circuit = transpile(grover_circuit, backend)
#shots = 1024
#result = backend.run(compiled_circuit, shots=shots).result()
#counts = result.get_counts()
#results[m] = counts
results = {}

for m in range(1, 6):
    grover_circuit = create_grover_circuit(m, n, k, U)

    grover_circuit.measure_all()
    compiled = transpile(grover_circuit, backend)

    result = backend.run(compiled, shots=1024).result()
    counts = result.get_counts()

    results[m] = counts

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, m in enumerate(range(1, 6)):
    ax = axes[i]
    counts = results[m]

    ax.bar(counts.keys(), counts.values())
    ax.set_title(f"m = {m}")
    ax.set_xlabel("State")
    ax.set_ylabel("Counts")
    ax.tick_params(axis='x', rotation=90)

# remove empty subplot (6th)
axes[-1].axis("off")

fig.suptitle("Observed Distribution for Grover's Algorithm (1024 shots) for m = 1 to 5")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Plot the results
#plt.bar(counts.keys(), counts.values())
#plt.xlabel('Measured State')
#plt.ylabel('Frequency')
#plt.title(f"Observed Distribution for Grover\'s Algorithm (1024 shots) with m=1")
#plt.grid(True)
#plt.show()

#measured_str = ''.join(reversed(max(counts, key=counts.get)))

#measured_int = int(measured_str, 2)
#print("Register Output = %i" % measured_int)
# Plot the observed distribution
# plot_results(counts)
#print(qiskit.__version__)
