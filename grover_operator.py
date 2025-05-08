#initialization
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

def create_grover_circuit(n, k, U):
    N = n**2
    assert k < N, "k must be less than N"
    #calculate ideal number of iterations m
    m = int(np.pi/4 * np.sqrt(N/k)-1/2)
    
    # Choose k good states randomly
    good_states = np.random.choice(N, k, replace=False)
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
n = 3  # Total number of states
k = 2  # Number of good states

U = QuantumCircuit(n)
U.h(range(n))

# Simulation for multiple iterations
backend = Aer.get_backend('qasm_simulator')
results = {}

grover_circuit = create_grover_circuit(n,k,U)

grover_circuit.measure_all()
compiled_circuit = transpile(grover_circuit, backend)
shots = 1024
result = backend.run(compiled_circuit, shots=shots).result()
counts = result.get_counts()
#results[m] = counts


# Plot the results
plt.bar(counts.keys(), counts.values())
plt.xlabel('Measured State')
plt.ylabel('Frequency')
plt.title(f"Observed Distribution for Grover\'s Algorithm (1024 shots) with m=1")
plt.grid(True)
plt.show()

# Plot the observed distribution
# plot_results(counts)
