from QAE import QAE_circuit
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

# Example Usage
n = 2  # Number of qubits
k_1 = 1
k_2 = 2  # Number of good states
t = 7  # Number of counting qubits

U = QuantumCircuit(n)
U.h(range(n))

QAE_1 = QAE_circuit(n,k_1,t,U)

QAE_1.measure(range(t), reversed(range(t)))

QAE_2 = QAE_circuit(n,k_2,t,U)

QAE_2.measure(range(t), (range(t)))

backend = Aer.get_backend('qasm_simulator')
results = {}

compiled_circuit_1 = transpile(QAE_1, backend)
compiled_circuit_2 = transpile(QAE_2, backend)
shots = 1024
result_1 = backend.run(compiled_circuit_1, shots=shots).result()
counts_1 = result_1.get_counts()
result_2 = backend.run(compiled_circuit_2, shots=shots).result()
counts_2 = result_2.get_counts()

# Plot the results
plt.bar(counts_1.keys(), counts_1.values())
plt.xlabel('Measured State')
plt.ylabel('Frequency')
plt.title(f"Observed Distribution for QAE (1024 shots) with k = 1")
plt.grid(True)
plt.show()

plt.bar(counts_2.keys(), counts_2.values())
plt.xlabel('Measured State')
plt.ylabel('Frequency')
plt.title(f"Observed Distribution for QAE (1024 shots) with k = 2")
plt.grid(True)
plt.show()

measured_str_1 = max(counts_1, key=counts_1.get)

measured_int_1 = int(measured_str_1, 2)
print("Register Output for k = 1: %i" % measured_int_1)

amplitude_1 = np.sin(np.pi*measured_int_1/2**t)**2
print("Estimated amplitude for k = 1: ", amplitude_1)

measured_str_2 = max(counts_2, key=counts_2.get)

measured_int_2 = int(measured_str_2, 2)
print("Register Output for k = 2: %i" % measured_int_2)

amplitude_2 = np.sin(np.pi*measured_int_2/2**t)**2
print("Estimated amplitude for k = 2: ", amplitude_2)

k_list = ['k = 1', 'k = 2']
amplitudes = [amplitude_1, amplitude_2]

plt.bar(k_list, amplitudes)
plt.title('Amplitudes estimated')
plt.xlabel('Number of good states (k)')
plt.ylabel('Amplitude')
plt.show()
