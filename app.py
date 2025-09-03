# Install dependencies before running this app
# !pip install streamlit qiskit plotly

import streamlit as st
from qiskit import QuantumCircuit, execute
from qiskit.providers.basicaer import QasmSimulator
from qiskit.quantum_info import Statevector
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Helper Functions ------------------
def get_bloch_coords(statevector, qubit_index):
    """Return Bloch sphere coordinates (x, y, z) for a single qubit."""
    rho = statevector.to_operator().data
    qubit_rho = np.trace(rho.reshape([2, 2] * statevector.num_qubits), axis1=qubit_index*2+1, axis2=qubit_index*2)
    qubit_rho = qubit_rho / np.trace(qubit_rho)
    x = np.real(np.trace(qubit_rho @ np.array([[0, 1], [1, 0]])))
    y = np.real(np.trace(qubit_rho @ np.array([[0, -1j], [1j, 0]])))
    z = np.real(np.trace(qubit_rho @ np.array([[1, 0], [0, -1]])))
    return [x, y, z]

def plot_bloch_interactive(coords, qubit_label):
    """Create interactive Bloch sphere for given coordinates."""
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale='Blues')])
    fig.add_trace(go.Scatter3d(x=[0, coords[0]], y=[0, coords[1]], z=[0, coords[2]],
                               mode='lines+markers', marker=dict(size=5, color='red'),
                               line=dict(color='red', width=5)))
    fig.update_layout(scene=dict(xaxis=dict(title='X'),
                                 yaxis=dict(title='Y'),
                                 zaxis=dict(title='Z')),
                      title=f"Bloch Sphere for Qubit {qubit_label}")
    return fig

def plot_statevector(statevector):
    """Plot statevector amplitudes."""
    amps = np.abs(statevector.data)**2
    fig, ax = plt.subplots()
    ax.bar(range(len(amps)), amps)
    ax.set_xlabel("Basis State")
    ax.set_ylabel("Probability")
    ax.set_title("Statevector Probabilities")
    return fig

def plot_histogram(qc):
    """Plot measurement histogram using Basic QASM simulator."""
    from qiskit.providers.basicaer import QasmSimulator
    backend = QasmSimulator()
    qc_measure = qc.copy()
    qc_measure.measure_all()
    job = backend.run(qc_measure, shots=1024)
    counts = job.result().get_counts()
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set_xlabel("States")
    ax.set_ylabel("Counts")
    ax.set_title("Measurement Histogram")
    return fig


# ----------------- Streamlit App ----------------------
st.title("Quantum Circuit Visualizer")
st.write("An interactive tool for building circuits and visualizing qubits for educational and research purposes.")

# Sidebar options
st.sidebar.header("Options")
input_type = st.sidebar.radio("Choose Input Type:", ["Predefined Circuit", "Upload QASM File", "Build Your Own"])

# Predefined circuits
predefined_options = {
    "Hadamard (H) Gate": QuantumCircuit(1).h(0),
    "CNOT Gate": QuantumCircuit(2),
    "Bell State": QuantumCircuit(2)
}
predefined_options["CNOT Gate"].h(0)
predefined_options["CNOT Gate"].cx(0, 1)
predefined_options["Bell State"].h(0)
predefined_options["Bell State"].cx(0, 1)

# User circuit initialization
qc = None
num_qubits = st.sidebar.number_input("Number of Qubits:", min_value=1, max_value=5, value=2)

if input_type == "Predefined Circuit":
    choice = st.sidebar.selectbox("Select a circuit:", list(predefined_options.keys()))
    qc = predefined_options[choice]

elif input_type == "Upload QASM File":
    uploaded_file = st.file_uploader("Upload your QASM file", type=["qasm"])
    if uploaded_file is not None:
        qasm_str = uploaded_file.read().decode("utf-8")
        qc = QuantumCircuit.from_qasm_str(qasm_str)
    else:
        st.warning("Please upload a valid QASM file to proceed.")
        st.stop()

else:  # Real-time circuit editing
    st.sidebar.subheader("Build Your Circuit")
    if "user_qc" not in st.session_state:
        st.session_state.user_qc = QuantumCircuit(num_qubits)
    
    gate = st.sidebar.selectbox("Select a Gate", ["H", "X", "Y", "Z", "CX"])
    target_qubit = st.sidebar.number_input("Target Qubit", 0, num_qubits-1, 0)

    if gate == "CX":
        control_qubit = st.sidebar.number_input("Control Qubit", 0, num_qubits-1, 0)
    
    if st.sidebar.button("Apply Gate"):
        if gate == "H":
            st.session_state.user_qc.h(target_qubit)
        elif gate == "X":
            st.session_state.user_qc.x(target_qubit)
        elif gate == "Y":
            st.session_state.user_qc.y(target_qubit)
        elif gate == "Z":
            st.session_state.user_qc.z(target_qubit)
        elif gate == "CX":
            st.session_state.user_qc.cx(control_qubit, target_qubit)

    qc = st.session_state.user_qc

# Show the circuit
if qc is not None:
    st.subheader("Quantum Circuit:")
    st.code(qc.draw(output="text"))

    # Get statevector
    sv = Statevector.from_instruction(qc)

    # Multiple visualizations
    st.subheader("Statevector Probabilities")
    fig_probs = plot_statevector(sv)
    st.pyplot(fig_probs)

    st.subheader("Measurement Histogram")
    fig_hist = plot_histogram(qc)
    st.pyplot(fig_hist)

    st.subheader("Bloch Sphere Representations (Interactive)")
    for qubit_index in range(qc.num_qubits):
        coords = get_bloch_coords(sv, qubit_index)
        fig_bloch = plot_bloch_interactive(coords, qubit_index)
        st.plotly_chart(fig_bloch, use_container_width=True)

    st.success("Visualization complete! Explore all representations interactively.")


