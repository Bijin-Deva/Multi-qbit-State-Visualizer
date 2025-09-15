# -*- coding: utf-8 -*-
"""
A Streamlit web application for visualizing quantum circuits from .qasm files,
updated with a modern, dark-themed interface and enhanced visualizations.
"""

import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import Aer
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt

# --- UPDATED: In-built QASM examples ---
EXAMPLES = {
    "Bell State (2 Qubits Entangled)": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
    """,
    "GHZ State (3 Qubits Entangled)": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0],q[1];
        cx q[0],q[2];
        measure q -> c;
    """,
    "Full Superposition (3 Qubits)": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        h q[1];
        h q[2];
        measure q -> c;
    """
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

def plot_bloch_sphere(x: float, y: float, z: float, title: str) -> go.Figure:
    """
    Generates a vibrant, interactive Bloch sphere plot.
    """
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.2, showscale=False,
        colorscale=[[0, 'rgb(30,30,60)'], [1, 'rgb(120,70,150)']],
        surfacecolor=np.sqrt(sphere_x**2 + sphere_y**2)
    ))
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines+text', text=['', 'X'], line=dict(color='#FF6666', width=4), textfont_color='#FF6666'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines+text', text=['', 'Y'], line=dict(color='#66FF66', width=4), textfont_color='#66FF66'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines+text', text=['|1⟩', '|0⟩'], line=dict(color='#6666FF', width=4), textfont_color='#6666FF'))
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines', line=dict(color='cyan', width=8), name='State Vector'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers', marker=dict(size=6, color='cyan', line=dict(width=2, color='white')), name='State'
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(color='white')),
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False, range=[-1.5, 1.5]),
            yaxis=dict(showticklabels=False, visible=False, range=[-1.5, 1.5]),
            zaxis=dict(showticklabels=False, visible=
