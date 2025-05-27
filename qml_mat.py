import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import concurrent.futures
import time # Optional: to measure execution time

# ----- Build 1D Transverse Field Ising Hamiltonian -----
def build_ising_hamiltonian(n, J, h):
    paulis = []
    coeffs = []

    # Z_i Z_{i+1} term
    for i in range(n - 1):
        label = ['I'] * n
        label[i] = 'Z'
        label[i + 1] = 'Z'
        paulis.append(''.join(reversed(label)))  # Qiskit uses right-to-left order
        coeffs.append(-J)

    # X_i term
    for i in range(n):
        label = ['I'] * n
        label[i] = 'X'
        paulis.append(''.join(reversed(label)))
        coeffs.append(-h)

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

def create_physically_inspired_ansatz(n_qubits, h, J, reps=3):
    params = ParameterVector('θ', length=2 * n_qubits * reps)
    qc = QuantumCircuit(n_qubits)

    # Initial state preparation: set RY angles according to h/J to simulate magnetization direction
    init_angle = 2 * np.arctan(h / J)  # Multiply by 2 because RY(θ)|0> = cos(θ/2)|0> + sin(θ/2)|1>
    for i in range(n_qubits):
        qc.ry(init_angle, i)

    # Ansatz part: multi-layer rotation + entanglement structure
    for rep in range(reps):
        for i in range(n_qubits):
            qc.ry(params[2 * n_qubits * rep + i], i)
            qc.rx(params[2 * n_qubits * rep + n_qubits + i], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc, list(params)

# Define h_list and N_list globally or pass them as arguments if preferred
h_list = np.logspace(-2, 2, 30)
N_list = range(2, 11)

def compute_mx_vs_h_for_N(n_qubits):
    # This warning is expected from Qiskit and can be managed if needed
    # import warnings
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    mx_list = []
    for h in h_list:
        J = 1.0
        hamiltonian = build_ising_hamiltonian(n_qubits, J, h)
        # Using reps=4 as in the notebook for this specific calculation
        ansatz_circuit, ansatz_params = create_physically_inspired_ansatz(n_qubits, h, J, reps=4) 
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=ansatz_circuit,
            observables=hamiltonian,
            input_params=[],
            weight_params=ansatz_params,
            estimator=estimator
        )
        qnn_model = TorchConnector(qnn)
        optimizer = torch.optim.AdamW(qnn_model.parameters(), lr=0.01)
        for epoch in range(200): # Reduced epochs for speed as in the notebook
            optimizer.zero_grad()
            output = qnn_model()
            loss = output.mean()
            loss.backward()
            optimizer.step()
        final_weights = qnn_model.weight.detach().numpy()
        x_obs = SparsePauliOp.from_list([(f"{'I'*i + 'X' + 'I'*(n_qubits - i - 1)}", 1.0) for i in range(n_qubits)])
        # Recreate estimator for thread safety if running in truly parallel Python threads, 
        # though ProcessPoolExecutor mitigates this by using separate processes.
        # For Estimator primitive, it should generally be fine.
        mx = Estimator().run(circuits=ansatz_circuit, observables=x_obs, parameter_values=[final_weights]).result().values[0] / n_qubits
        mx_list.append(mx)
    print(f"Finished N={n_qubits}")
    return mx_list

if __name__ == '__main__':
    start_time = time.time() # Optional: for timing

    mx_vs_h = np.zeros((len(N_list), len(h_list)))
    
    # Using ProcessPoolExecutor for CPU-bound tasks like these Qiskit simulations
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map will submit tasks and return results in the order of the input iterable N_list
        results = list(executor.map(compute_mx_vs_h_for_N, N_list))
        
        for iN, mx_list_result in enumerate(results):
            mx_vs_h[iN, :] = mx_list_result

    plt.figure(figsize=(8,6))
    for iN, N_val in enumerate(N_list): # Use N_val to avoid conflict with numpy N
        plt.plot(h_list, mx_vs_h[iN], label=f"N={N_val}")
    plt.xscale('log')
    plt.xlabel('Transverse Field h (log scale)')
    plt.ylabel(r'$\langle X \rangle$')
    plt.title(r'QML: Magnetization $\langle X \rangle$ vs $h$ for $N=2$ to $10$ (J=1)')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    
    end_time = time.time() # Optional: for timing
    print(f"Total execution time: {end_time - start_time:.2f} seconds") # Optional

    plt.savefig("magnetization_vs_h_multiprocess.png") # Save the plot
    plt.show()
    print("Plot saved as magnetization_vs_h_multiprocess.png")