# -*- coding: utf-8 -*-
"""
A Streamlit web application for visualizing quantum circuits from .qasm files,
updated with a modern, light-themed (cream/yellow) interface and enhanced visualizations.
"""

import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import Aer
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt

# --- UPDATED: Examples now include a 'note' for the UI ---
EXAMPLES = {
    "Single Qubit Superposition (Vector Visible)": { # <<< ADDED THIS EXAMPLE
        "qasm": """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            measure q -> c;
        """,
        "note": """
        **Circuit Explanation:** This circuit puts a single qubit into a $|+\rangle$ state (a superposition of 0 and 1).
        Because this qubit is **not entangled** with any other, it has a "pure state" that can be drawn on the sphere.
        You should see the pink vector pointing directly to the **'X' axis**, at position `(1, 0, 0)`. The Purity will be `1.0`.
        """
    },
    "Bell State (Entanglement)": {
        "qasm": """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            measure q -> c;
        """,
        "note": """
        **Circuit Explanation:** This circuit creates a Bell state, a fundamental example of quantum entanglement.
        The Hadamard gate puts the first qubit in a superposition. The CNOT gate then entangles the second qubit with the first.
        Because of this entanglement, the measurement outcomes of the two qubits are perfectly correlated. You will only ever measure **`00`** or **`11`**, each with roughly 50% probability.
        """
    },
    "GHZ State (3-Qubit Entanglement)": {
        "qasm": """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0],q[1];
            cx q[0],q[2];
            measure q -> c;
        """,
        "note": """
        **Circuit Explanation:** The Greenberger–Horne–Zeilinger (GHZ) state is an entangled state of three qubits.
        The first qubit is put into a superposition, and then CNOT gates are used to entangle the other two qubits with it.
        The result is that all three qubits are linked. You will only ever measure **`000`** or **`111`**, each with roughly 50% probability.
        """
    },
    "Full Superposition (3 Qubits)": {
        "qasm": """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            h q[1];
            h q[2];
            measure q -> c;
        """,
        "note": """
        **Circuit Explanation:** Applying a Hadamard (H) gate to every qubit puts the entire system into an equal superposition of all possible basis states.
        For 3 qubits, there are $2^3 = 8$ possible outcomes (from `000` to `111`). When you measure the circuit, each of these 8 outcomes has an equal probability of occurring.
        """
    }
}


# --- Pauli Matrices (Constants) ---
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# --- Core Quantum Calculation Functions ---

def remove_final_measurements_if_any(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc without final measurements."""
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instr, qargs, cargs in qc.data:
        if instr.name != "measure":
            new_qc.append(instr, qargs, cargs)
    return new_qc

def statevector_from_circuit(qc: QuantumCircuit) -> Statevector:
    """Calculates the statevector from a quantum circuit."""
    qc2 = remove_final_measurements_if_any(qc)
    return Statevector.from_instruction(qc2)

def reduced_density_for_qubit(state: Statevector, target_qubit: int) -> np.ndarray:
    """Traces out all qubits except the target_qubit to get its density matrix."""
    return partial_trace(state, [q for q in range(state.num_qubits) if q != target_qubit]).data

def bloch_vector_from_rho(rho2x2: np.ndarray):
    """Calculates the Bloch vector (x, y, z) from a 2x2 density matrix."""
    x = np.real(np.trace(rho2x2 @ SX))
    y = np.real(np.trace(rho2x2 @ SY))
    z = np.real(np.trace(rho2x2 @ SZ))
    return float(x), float(y), float(z)

def purity_from_rho(rho2x2: np.ndarray):
    """Calculates the purity of the state from its density matrix."""
    return float(np.real(np.trace(rho2x2 @ rho2x2)))

# --- Visualization Function ---

# ##################################################
# # --- UPDATED: plot_bloch_sphere FUNCTION for better visibility ---
# ##################################################
def plot_bloch_
