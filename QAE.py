#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit_aer import Aer
from qiskit import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT

# import basic plot tools
from qiskit.visualization import plot_histogram

def grover_operator(n, k, U, n_iterations, good_states):
    # Construct the oracle that marks good states
    oracle = QuantumCircuit(n, global_phase=np.pi)
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
                
    # Convert U to gate and inverse
    U_gate = U.to_gate()
    U_inv_gate = U_gate.inverse()
    
    # Construct the diffusion operator
    diffuser = QuantumCircuit(n)
    # Start with the conjugate transpose of the state preparation (U†)
    diffuser.append(U_inv_gate, range(n))
    # Apply the reflection around |0⟩
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    # Reapply the state preparation (U)
    diffuser.append(U_gate, range(n))
    
    # Combine the Grover operator
    grover_op = oracle.compose(diffuser)
    # Create an empty circuit to hold repeated Grover iterations
    grover_it = QuantumCircuit(n)
    #grover_it.append(U_gate,range(n))

    # Compose grover_op n_iterations times
    for _ in range(n_iterations):
        grover_it.compose(grover_op, inplace=True)

    # Convert the repeated Grover operator to a gate
    grover_gate = grover_it.to_gate()
    grover_gate.name = f"Grover^{n_iterations}"
    
    return grover_gate


def QAE_circuit(n,k,t,U):
    N = 2**n
    assert k < N, "k must be less than N"
    qc = QuantumCircuit(n+t,t)

    qft_dagger = QFT(t).inverse().to_gate()
    qft_dagger.label = "QFT†"
    
    # Choose k good states randomly
    #good_states = np.random.choice(N, k, replace=False)
    if k==1:
        good_states = np.array([3])
    if k==4:
        good_states = np.array([3,5,2,1])
    #good_states = np.array([3, 5])  # fix to test known states
    #print(f'good states: {good_states}')
    
    for qubit in range(t):
        qc.h(qubit)
    
    U_gate = U.to_gate()
    U_gate.name = "Init_op"
    qc.append(U_gate,range(t,t+n))

    # Begin controlled Grover iterations
    n_iterations = 1
    for qubit in range(t):
        cgrit = grover_operator(n,k,U,n_iterations,good_states).control()
        qc.append(cgrit, [qubit] + list(range(t, n+t)))
        n_iterations *= 2
    
    # Do inverse QFT on counting qubits
    qc.append(qft_dagger, range(t))

    return qc

if __name__ == "__main__":
    # Example Usage
    n = 1  # Number of qubits
    k = 2  # Number of good states
    t = 3  # Number of counting qubits

    U = QuantumCircuit(n)
    U.h(range(n))

    QAE = QAE_circuit(n,k,t,U)

    QAE.measure(range(t), range(t))
    QAE.draw(fold=-1)



    backend = Aer.get_backend('qasm_simulator')
    results = {}

    compiled_circuit = transpile(QAE, backend)
    shots = 1024
    result = backend.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    # Plot the results
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Measured State')
    plt.ylabel('Frequency')
    plt.title(f"Observed Distribution for QAE (1024 shots)")
    plt.grid(True)
    plt.show()

    measured_str = max(counts, key=counts.get)

    measured_int = int(measured_str, 2)
    print("Register Output = %i" % measured_int)

    amplitude = np.sin(np.pi*measured_int/(2**t))**2
    print("Estimated amplitude = ", amplitude)

