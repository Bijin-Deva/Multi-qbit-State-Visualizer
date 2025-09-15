# -*- coding: utf-8 -*-
"""
A Streamlit web application for visualizing quantum circuits from .qasm files,
updated with a modern, dark-themed interface and enhanced visualizations.
This version adds support for viewing measurement gates and initializing
custom quantum states like |10>.
"""

import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import Aer
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt

# --- QASM examples with measurements and custom states ---
EXAMPLES = {
    "Bell State (Entanglement)": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
    """,
    "GHZ State (3-Qubit Entanglement)": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0],q[1];
        cx q[0],q[2];
        measure q -> c;
    """,
    "Superposition & Measurement": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    """,
    "Initialized State |10>": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[1]; // Flip q_1 from |0> to |1>. State is |10>
        measure q -> c;
    """
}

# --- Pauli Matrices (Constants) ---
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# --- Core Quantum Calculation Functions ---

def remove_final_measurements_if_any(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc without final measurements for statevector analysis."""
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
    Generates an interactive Bloch sphere plot.
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
st.markdown("Choose an example or upload a **`.qasm`** file to visualize a quantum circuit.")

# --- Sidebar with example selection ---
st.sidebar.title("Circuit Source")
example_options = list(EXAMPLES.keys())
choice = st.sidebar.selectbox(
    "Choose an example or upload your own",
    ["Select an option..."] + example_options + ["Upload my own..."]
)

qasm_text = None
uploaded_file = None

if choice in example_options:
    qasm_text = EXAMPLES[choice]
elif choice == "Upload my own...":
    uploaded_file = st.sidebar.file_uploader("Choose a .qasm file", type="qasm")
    if uploaded_file is not None:
        qasm_text = io.BytesIO(uploaded_file.getvalue()).read().decode("utf-8")

st.sidebar.title("Simulation Controls")
num_shots = st.sidebar.slider('Number of Shots (for measurement)', 100, 8192, 1024)

# --- Main application logic ---
if qasm_text is not None:
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_text)

        st.header("Quantum Circuit")
        st.markdown("This diagram shows the gates and measurements as defined in the QASM file.")
        
        # Define a custom style dictionary to ensure all text is visible and layout is better
        custom_style = {
            "gatefacecolor": "#3B5998",
            "gatetextcolor": "white",
            "linecolor": "#AAAAAA",
            "textcolor": "white",
            "labelcolor": "white",
            "creg_labelfontsize": 11,
            "qreg_labelfontsize": 11,
            "fontsize": 9,
            "dpi": 200,
            "margin": [0.1, 0.01, 0.01, 0.05]  # [left, bottom, right, top]
        }
        # Create the figure and make its background transparent
        fig, ax = plt.subplots(figsize=(6, max(2.0, qc.num_qubits * 0.45)))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Draw the circuit using the new custom style
        qc.draw(output='mpl', style=custom_style, ax=ax)
        
        st.pyplot(fig)

        # --- Measurement Simulation ---
        with st.spinner("Simulating measurements..."):
            st.header("Classical Measurement Outcomes")
            
            qc_for_measurement = qc.copy()
            
            if qc_for_measurement.num_clbits == 0 and qc_for_measurement.num_qubits > 0:
                st.info("No classical registers found in QASM. Adding measurements to all qubits for simulation.")
                qc_for_measurement.measure_all(inplace=True)

            if qc_for_measurement.num_clbits > 0:
                qasm_backend = Aer.get_backend('qasm_simulator')
                qasm_job = qasm_backend.run(qc_for_measurement, shots=num_shots)
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
                    most_likely_outcome = max(counts, key=counts.get)
                    st.subheader("Most Probable Outcome")
                    st.markdown(f"### `{most_likely_outcome}`")
            else:
                st.warning("Circuit has no measurement operations. No outcomes to display.")

        # --- Ideal Quantum State Analysis ---
        with st.spinner("Calculating ideal quantum states..."):
            state = statevector_from_circuit(qc)

            st.header("Qubit State Analysis")
            st.markdown("Below is the analysis for each individual qubit's state *before* measurement, calculated by tracing out all other qubits.")

            if qc.num_qubits > 0:
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
            else:
                st.info("No qubits in the circuit to analyze.")


    except Exception as e:
        st.error(f"An error occurred while processing the QASM file: {e}")
        st.warning("Please ensure the QASM is valid and that your environment includes qiskit-aer (`pip install qiskit-aer`).")
else:
    st.info("Please select an example or upload a .qasm file using the sidebar to begin.")
