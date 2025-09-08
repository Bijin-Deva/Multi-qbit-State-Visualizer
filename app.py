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

# --- Pauli Matrices (Constants) ---
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# --- Core Quantum Calculation Functions ---

def remove_final_measurements_if_any(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc without final measurements."""
    try:
        return qc.remove_final_measurements(inplace=False)
    except Exception:
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

# --- Updated Visualization Function ---

def plot_bloch_sphere(x: float, y: float, z: float, title: str) -> go.Figure:
    """
    Generates a vibrant, interactive Bloch sphere plot with uniquely colored axes.
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
            zaxis=dict(showticklabels=False, visible=False, range=[-1.5, 1.5]),
            aspectmode='cube'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# --- Streamlit User Interface ---

st.set_page_config(page_title="Quantum Circuit Visualizer", layout="wide")

st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(to bottom right, #000000, #0D224F);
    background-attachment: fixed;
    background-size: cover;
}
[data-testid="stHeader"] { background-color: transparent; }
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] .stMarkdown p { color: white !important; }
[data-testid="stSidebar"] { background-color: #0A193D; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }
[data-testid="stMetric"] label,
[data-testid="stMetric"] div { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ Quantum Circuit Visualizer")
st.markdown("Upload a quantum circuit in **OpenQASM 2.0 (`.qasm`)** format to see the state of each qubit visualized on a Bloch sphere.")

st.sidebar.title("Circuit Upload")
uploaded_file = st.sidebar.file_uploader("Choose a .qasm file", type="qasm")

st.sidebar.title("Simulation Controls")
num_shots = st.sidebar.slider('Number of Shots (for measurement)', 100, 8192, 1024)


if uploaded_file is not None:
    qasm_text = io.BytesIO(uploaded_file.getvalue()).read().decode("utf-8")
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_text)

        st.header("Uploaded Quantum Circuit")
        fig, ax = plt.subplots(figsize=(8, max(2, qc.num_qubits * 0.5)))
        qc.draw(output='mpl', style='iqp', ax=ax)
        st.pyplot(fig)

        with st.spinner("Simulating measurements..."):
            st.header("Classical Measurement Outcomes")
            
            qc_measured = qc.copy()
            qc_measured.measure_all(inplace=True) 
            
            qasm_backend = Aer.get_backend('qasm_simulator')
            qasm_job = qasm_backend.run(qc_measured, shots=num_shots)
            counts = qasm_job.result().get_counts()
            
            sorted_counts = dict(sorted(counts.items()))

            hist_fig = go.Figure(go.Bar(
                x=list(sorted_counts.keys()), 
                y=list(sorted_counts.values()),
                marker_color='indianred'
            ))
            hist_fig.update_layout(
                title=dict(text=f"Results from {num_shots} shots", font_color='white'),
                xaxis_title="Outcome (Classical Bit String)",
                yaxis_title="Counts",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.2)',
                font_color='white'
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            if counts:
                # Find the outcome with the highest count
                most_likely_outcome_raw = max(counts, key=counts.get)
                
                # --- FIX STARTS HERE ---
                # Clean the string by removing any spaces
                final_outcome = most_likely_outcome_raw.replace(" ", "")
                
                st.subheader("Most Probable Outcome")
                st.markdown(f"### `{final_outcome}`")
                # --- FIX ENDS HERE ---
            else:
                st.warning("No measurement outcomes were recorded.")

        with st.spinner("Calculating ideal quantum states..."):
            state = statevector_from_circuit(qc)

            st.header("Qubit State Analysis")
            st.markdown("Below is the analysis for each individual qubit after tracing out all others.")

            cols = st.columns(qc.num_qubits)
            for i in range(qc.num_qubits):
                with cols[i]:
                    rho = reduced_density_for_qubit(state, i)
                    bx, by, bz = bloch_vector_from_rho(rho)
                    p = purity_from_rho(rho)
                    
                    fig_bloch = plot_bloch_sphere(bx, by, bz, title=f"Qubit {i}")
                    st.plotly_chart(fig_bloch, use_container_width=True)
                    
                    st.metric(label=f"Purity (Qubit {i})", value=f"{p:.4f}")
                    
                    with st.expander(f"Details for Qubit {i}"):
                        st.markdown(f"**Bloch Vector:** `({bx:.3f}, {by:.3f}, {bz:.3f})`")
                        st.markdown("Reduced Density Matrix:")
                        st.dataframe(np.round(rho, 3))

    except Exception as e:
        st.error(f"An error occurred while processing the QASM file: {e}")
        st.warning("Please ensure the uploaded file is a valid OpenQASM 2.0 file and that your environment includes qiskit-aer (`pip install qiskit-aer`).")
else:
    st.info("Awaiting a .qasm file. Please upload a circuit using the sidebar.")
