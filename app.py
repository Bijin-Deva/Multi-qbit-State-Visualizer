# -*- coding: utf-8 -*-
"""
A Streamlit web application for visualizing quantum circuits on the Bloch sphere.

This app allows users to upload a quantum circuit defined in a .qasm file.
It then calculates the statevector for the circuit and, for each qubit,
computes and displays its Bloch vector, purity, and an interactive 3D
representation on the Bloch sphere using Plotly.
"""

import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
import plotly.graph_objects as go
import io

# --- Pauli Matrices (Constants) ---
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# --- Core Functions from your Notebook ---
# These functions perform the quantum calculations.

def remove_final_measurements_if_any(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc without final measurements."""
    try:
        # Qiskit >= 0.45 has this built-in method
        return qc.remove_final_measurements(inplace=False)
    except Exception:
        # Fallback for older Qiskit versions
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
    total_qubits = int(np.log2(state.data.size))
    trace_out = [i for i in range(total_qubits) if i != target_qubit]
    return partial_trace(state, trace_out).data

def bloch_vector_from_rho(rho2x2: np.ndarray):
    """Calculates the Bloch vector (x, y, z) from a 2x2 density matrix."""
    x = np.real(np.trace(rho2x2 @ SX))
    y = np.real(np.trace(rho2x2 @ SY))
    z = np.real(np.trace(rho2x2 @ SZ))
    return float(x), float(y), float(z)

def purity_from_rho(rho2x2: np.ndarray):
    """Calculates the purity of the state from its density matrix."""
    return float(np.real(np.trace(rho2x2 @ rho2x2)))

def bloch_plotly(x, y, z, title="Bloch vector"):
    """
    Generates an interactive 3D Bloch sphere plot using Plotly.
    MODIFIED: This function now returns the figure object instead of showing it.
    """
    # Create a sphere surface for the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    # Add the semi-transparent sphere surface
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.2, showscale=False, colorscale=[[0, 'lightblue'], [1, 'lightblue']]
    ))

    # Add the X, Y, Z axes lines
    fig.add_trace(go.Scatter3d(x=[-1.1, 1.1], y=[0, 0], z=[0, 0], mode='lines+text', text=['-X', 'X'], line=dict(color='gray')))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.1, 1.1], z=[0, 0], mode='lines+text', text=['-Y', 'Y'], line=dict(color='gray')))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.1, 1.1], mode='lines+text', text=['|1⟩', '|0⟩'], line=dict(color='gray')))

    # Add the Bloch vector arrow
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines',
        line=dict(color='red', width=5),
        name='Bloch Vector'
    ))

    # Add a marker for the tip of the vector
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers',
        marker=dict(size=8, color='darkred'),
        name='Vector Tip'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(range=[-1, 1], showticklabels=False, title=''),
            yaxis=dict(range=[-1, 1], showticklabels=False, title=''),
            zaxis=dict(range=[-1, 1], showticklabels=False, title=''),
            aspectmode="cube"  # Ensures the sphere is not distorted
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    return fig

# --- Streamlit User Interface ---

st.set_page_config(page_title="Quantum Circuit Visualizer", layout="wide")

st.title("⚛️ Quantum Circuit Bloch Sphere Visualizer")
st.write("Upload a quantum circuit in OpenQASM 2.0 (`.qasm`) format to see the state of each qubit visualized on a Bloch sphere.")

# Sidebar for file upload
st.sidebar.header("Circuit Upload")
uploaded_file = st.sidebar.file_uploader("Choose a .qasm file", type="qasm")

if uploaded_file is not None:
    # Read the uploaded file content
    qasm_text = io.BytesIO(uploaded_file.getvalue()).read().decode("utf-8")

    try:
        # Create the QuantumCircuit from the QASM string
        qc = QuantumCircuit.from_qasm_str(qasm_text)

        st.header("Uploaded Quantum Circuit")
        # --- MODIFICATION START ---
        # Generate the circuit diagram as a matplotlib figure
        fig = qc.draw(output='mpl', style='iqp') 
        # 2. Set the desired size in inches (width, height)
        fig.set_size_inches(8, 3)
        # Display the figure in the Streamlit app
        st.pyplot(fig)
        # --- MODIFICATION END ---

        # Calculate the statevector for the entire circuit
        state = statevector_from_circuit(qc)

        st.header("Qubit State Analysis")
        st.write("Below is the analysis for each individual qubit after tracing out all others.")

        # Analyze and display each qubit
        for i in range(qc.num_qubits):
            st.markdown(f"---")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader(f"Qubit {i}")
                # Calculate properties for the specific qubit
                rho = reduced_density_for_qubit(state, i)
                bx, by, bz = bloch_vector_from_rho(rho)
                p = purity_from_rho(rho)

                # Display metrics
                st.metric(label="Purity", value=f"{p:.4f}")
                st.write("**Bloch Vector Coordinates:**")
                st.code(f"x = {bx:.4f}\ny = {by:.4f}\nz = {bz:.4f}", language='text')

            with col2:
                # Generate and display the Bloch sphere plot
                fig_bloch = bloch_plotly(bx, by, bz, title=f"Bloch Sphere for Qubit {i}")
                st.plotly_chart(fig_bloch, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the QASM file: {e}")
        st.warning("Please ensure the uploaded file is a valid OpenQASM 2.0 file.")
else:
    st.info("Awaiting a .qasm file. Please upload a circuit using the sidebar.")




