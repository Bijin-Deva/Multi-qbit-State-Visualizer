# -*- coding: utf-8 -*-
"""
A Streamlit web application for visualizing quantum circuits from .qasm files,
updated with a modern, light-themed (cream/yellow) interface, enhanced visualizations,
and a new noise simulation feature.
"""

import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix # <<< ADDED DensityMatrix
from qiskit_aer import AerSimulator # <<< CHANGED from Aer
from qiskit_aer.noise import NoiseModel, bit_flip, phase_flip, depolarizing_error # <<< NEW
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt

# --- UPDATED: Examples now include a 'note' for the UI ---
EXAMPLES = {
    "Arbitrary State (In-Between Axes)": {
        "qasm": """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            ry(0.785398) q[0];
            measure q -> c;
        """,
        "note": """
        **Circuit Explanation:** This circuit is designed to show a state *in between* the main axes.
        The H gate puts the qubit on the X-axis, and the ry($\pi/4$) gate rotates it.
        The result is a pink vector that points diagonally, in between the X and Z axes.
        """
    },
    "Single Qubit Superposition (Vector Visible)": {
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
        Because this qubit is **not entangled**, it has a "pure state" (Purity=1.0) and the vector is long.
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
        **Circuit Explanation:** This circuit creates a Bell state. Because the qubits are entangled,
        the individual qubit states are "maximally mixed" (Purity=0.5) and the vector is at (0,0,0).
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
        Like the Bell state, the individual qubits are in a mixed state (Purity=0.5).
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
        **Circuit Explanation:** Applying a Hadamard (H) gate to every qubit puts the entire system into an equal superposition of all 8 possible outcomes.
        The individual qubits are *not* entangled and are all in the $|+\rangle$ state (Purity=1.0).
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

# ##################################################
# # --- NEW: Noise Model Helper Function ---
# ##################################################
def create_noise_model(noise_type: str, probability: float) -> NoiseModel:
    """Creates a Qiskit noise model based on user selection."""
    noise_model = NoiseModel()
    
    # List of gates to apply noise to (all except measure/barrier)
    gate_list = ['h', 'x', 'y', 'z', 'cx', 'ry', 'sx'] 
    
    if noise_type == "Bit Flip Error":
        error = bit_flip(probability)
        noise_model.add_all_qubit_quantum_error(error, gate_list)
        
    elif noise_type == "Phase Flip Error":
        error = phase_flip(probability)
        noise_model.add_all_qubit_quantum_error(error, gate_list)
        
    elif noise_type == "Depolarizing Error":
        # 1-qubit depolarizing error
        error_1 = depolarizing_error(probability, 1)
        # 2-qubit depolarizing error (applied to 2-qubit gates)
        error_2 = depolarizing_error(probability, 2).power(1/2) # Scale error for CNOT
        
        noise_model.add_all_qubit_quantum_error(error_1, [g for g in gate_list if g != 'cx'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        
    return noise_model
# ##################################################


# --- Visualization Function ---
def plot_bloch_sphere(x: float, y: float, z: float, title: str) -> go.Figure:
    """
    Generates an interactive Bloch sphere plot, styled for a light theme
    and mimicking the appearance of Image 2, with a more visible state vector.
    """
    # Create the sphere mesh
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    # Add the translucent sphere surface
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.15,
        showscale=False,
        colorscale='Greys',
        surfacecolor=np.sqrt(sphere_x**2 + sphere_y**2),
        lighting=dict(ambient=0.8, diffuse=0.1, specular=0.1)
    ))

    # Add grid lines (parallels and meridians)
    for i in range(0, 360, 30):
        u_grid = np.deg2rad(i)
        grid_x = np.cos(u_grid) * np.sin(v)
        grid_y = np.sin(u_grid) * np.sin(v)
        grid_z = np.cos(v)
        fig.add_trace(go.Scatter3d(
            x=grid_x, y=grid_y, z=grid_z,
            mode='lines', line=dict(color='lightgrey', width=1), showlegend=False
        ))
    for i in range(0, 180, 30):
        v_grid = np.deg2rad(i)
        grid_x = np.cos(u) * np.sin(v_grid)
        grid_y = np.sin(u) * np.sin(v_grid)
        grid_z = np.cos(v_grid) * np.ones_like(u)
        fig.add_trace(go.Scatter3d(
            x=grid_x, y=grid_y, z=grid_z,
            mode='lines', line=dict(color='lightgrey', width=1), showlegend=False
        ))

    # Add axes with labels at the ends
    axis_length = 1.2
    axis_color = 'darkgrey'
    label_font_color = '#333333'

    fig.add_trace(go.Scatter3d(x=[-axis_length, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color=axis_color, width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[axis_length], y=[0], z=[0], mode='text', text=['X'], textfont=dict(color=label_font_color, size=14), showlegend=False))

    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-axis_length, axis_length], z=[0, 0], mode='lines', line=dict(color=axis_color, width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[axis_length], z=[0], mode='text', text=['Y'], textfont=dict(color=label_font_color, size=14), showlegend=False))

    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-axis_length, axis_length], mode='lines', line=dict(color=axis_color, width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_length], mode='text', text=['|0⟩'], textfont=dict(color=label_font_color, size=14), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-axis_length], mode='text', text=['|1⟩'], textfont=dict(color=label_font_color, size=14), showlegend=False))


    # Add the state vector as an arrow (using cone for arrowhead) and a marker at the tip
    arrow_color = '#FF1493' # Deep Pink
    vector_magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    # Only draw the vector if its length is significant
    if vector_magnitude > 0.05: # Threshold to avoid drawing tiny vectors at (0,0,0)
        # Vector line
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color=arrow_color, width=8),
            name='State Vector', showlegend=False
        ))
        unit_x, unit_y, unit_z = x/vector_magnitude, y/vector_magnitude, z/vector_magnitude
        
        # Arrowhead (cone)
        cone_size = 0.15
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[z],
            u=[unit_x], v=[unit_y], w=[unit_z],
            sizemode="absolute", sizeref=cone_size, anchor="tip",
            showscale=False,
            colorscale=[[0, arrow_color], [1, arrow_color]],
            showlegend=False
        ))
        # Add a marker sphere at the tip of the vector
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=7, color=arrow_color, line=dict(width=1, color='white')),
            name='State Point', showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(color='#333333')),
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False, range=[-1.5, 1.5]),
            yaxis=dict(showticklabels=False, visible=False, range=[-1.5, 1.5]),
            zaxis=dict(showticklabels=False, visible=False, range=[-1.5, 1.5]),
            aspectmode='cube',
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


# --- Streamlit User Interface ---

st.set_page_config(page_title="Quantum Circuit Visualizer", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #FFFBEA; /* Pale Yellow/Cream Background */
}
/* ... (rest of your CSS is unchanged) ... */
[data-testid="stHeader"] { background-color: transparent; }
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] .stMarkdown p { color: #333333 !important; } /* Dark text */
[data-testid="stSidebar"] { background-color: #FAF0E6; } /* Light cream/linen sidebar */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #333333 !important; } /* Dark text */
[data-testid="stMetric"] label,
[data-testid="stMetric"] div { color: #333333 !important; } /* Dark text */
[data-testid="stInfo"] { background-color: rgba(240, 230, 140, 0.3); } /* Light yellow info box */
[data-testid="stExpander"] summary {
    color: #004E98 !important; /* Dark readable blue */
    font-weight: bold;
}
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
note_text = None # Variable to hold the note for an example

if choice in example_options:
    qasm_text = EXAMPLES[choice]["qasm"]
    note_text = EXAMPLES[choice]["note"]
elif choice == "Upload my own...":
    uploaded_file = st.sidebar.file_uploader("Choose a .qasm file", type="qasm")
    if uploaded_file is not None:
        qasm_text = io.BytesIO(uploaded_file.getvalue()).read().decode("utf-8")

st.sidebar.title("Simulation Controls")
num_shots = st.sidebar.slider('Number of Shots (for measurement)', 100, 8192, 1024)

# ##################################################
# # --- NEW: Noise Control UI ---
# ##################################################
st.sidebar.subheader("Noise Controls")
noise_type = st.sidebar.radio(
    "Select Noise Model",
    ["No Noise (Ideal)", "Bit Flip Error", "Phase Flip Error", "Depolarizing Error"],
    help="Select a noise model to apply to the simulation. This simulates errors in real quantum hardware."
)
noise_level = st.sidebar.slider(
    "Noise Probability", 
    min_value=0.0, 
    max_value=0.5, 
    value=0.05,  # Default 5% noise
    step=0.01,
    disabled=(noise_type == "No Noise (Ideal)"),
    help="The probability (from 0% to 50%) that an error will occur on each gate."
)
# ##################################################


# --- Main app logic ---
if qasm_text is not None:
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_text)

        st.header("Quantum Circuit")
        st.markdown("This diagram shows the gates and measurements as defined in the QASM file.")

        # --- Matplotlib Circuit Drawing (Unchanged) ---
        custom_style = {
            "textcolor": "#333333", "gatetextcolor": "#000000", "labelcolor": "#333333",
            "linecolor": "#888888", "creglinecolor": "#888888", "gatefacecolor": "#ADD8E6",
            "barrierfacecolor": "#AAAAAA", "fontsize": 10,
            "displaycolor": {'h': '#87CEEB', 'cx': '#87CEEB', 'x': '#F08080', 'measure': '#808080'},
            "dpi": 200, "margin": [0.25, 0.1, 0.1, 0.1],
            "qreg_textalign": "left", "creg_textalign": "left"
        }
        fig, ax = plt.subplots(figsize=(8, max(2.5, qc.num_qubits * 0.6)))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.axis('off')
        qc.draw(output='mpl', style=custom_style, ax=ax, scale=0.7, initial_state=True)
        st.pyplot(fig)


        # ####################################################################
        # # --- NEW: Main Logic Split for Ideal vs. Noisy Simulation ---
        # ####################################################################

        if noise_type == "No Noise (Ideal)":
            # --- This is the ORIGINAL simulation logic ---
            
            # --- 1. Measurement Simulation ---
            with st.spinner("Simulating ideal measurements..."):
                st.header("Classical Measurement Outcomes (Ideal)")

                qc_for_measurement = qc.copy()
                if qc_for_measurement.num_clbits == 0 and qc_for_measurement.num_qubits > 0:
                    st.info("No classical registers found. Adding measurements to all qubits for simulation.")
                    qc_for_measurement.measure_all(inplace=True)

                if qc_for_measurement.num_clbits > 0:
                    # Use the standard ideal simulator
                    qasm_backend = AerSimulator() # Using AerSimulator is fine here too
                    qasm_job = qasm_backend.run(qc_for_measurement, shots=num_shots)
                    counts = qasm_job.result().get_counts()
                    sorted_counts = dict(sorted(counts.items()))

                    # ... (Histogram plotting logic - unchanged) ...
                    hist_fig = go.Figure(go.Bar(x=list(sorted_counts.keys()), y=list(sorted_counts.values()), marker_color='indianred'))
                    hist_fig.update_layout(title=dict(text=f"Results from {num_shots} shots", font_color='#333333'),
                                           xaxis_title="Outcome (Classical Bit String)", yaxis_title="Counts",
                                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(230, 230, 230, 0.2)', font_color='#333333')
                    st.plotly_chart(hist_fig, use_container_width=True)
                    # ... (Rest of histogram info logic - unchanged) ...
                    if counts:
                        most_likely_outcome = max(counts, key=counts.get)
                        st.subheader("Most Probable Outcome")
                        st.markdown(f"### `{most_likely_outcome}`")
                        with st.expander("Show Raw Counts"):
                            st.json(sorted_counts)
                        if qc.num_qubits > 0:
                            readout_order = "".join([f"q{i}" for i in range(qc.num_qubits - 1, -1, -1)])
                            st.info(f"💡 **How to Read the Output:** The bit string `{most_likely_outcome}` corresponds to the qubits in the order **`{readout_order}`**.")
                        if note_text:
                            st.markdown(note_text)
                else:
                    st.info("Circuit contains no classical registers for measurement outcomes.")

            # --- 2. Ideal Quantum State Analysis ---
            with st.spinner("Calculating ideal quantum states..."):
                state = statevector_from_circuit(qc)
                st.header("Qubit State Analysis (Ideal)")
                st.markdown("Below is the analysis for each individual qubit after tracing out all others.")

                if qc.num_qubits > 0:
                    cols = st.columns(qc.num_qubits)
                    for i in range(qc.num_qubits):
                        with cols[i]:
                            rho = reduced_density_for_qubit(state, i) # Uses ideal statevector
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
        
        else:
            # --- This is the NEW Noisy simulation logic ---
            st.info(f"Applying **{noise_type}** with **{noise_level*100:.1f}%** probability to gates.")

            # 1. Get the noise model and create the noisy simulator
            noise_model = create_noise_model(noise_type, noise_level)
            noisy_sim = AerSimulator(noise_model=noise_model)

            # --- 1. Noisy Measurement Simulation ---
            with st.spinner(f"Simulating measurements with {noise_type}..."):
                st.header(f"Classical Measurement Outcomes ({noise_type})")

                qc_for_measurement = qc.copy()
                if qc_for_measurement.num_clbits == 0 and qc_for_measurement.num_qubits > 0:
                    st.info("No classical registers found. Adding measurements to all qubits for simulation.")
                    qc_for_measurement.measure_all(inplace=True)

                if qc_for_measurement.num_clbits > 0:
                    # Run the simulation with the noisy simulator
                    qasm_job = noisy_sim.run(qc_for_measurement, shots=num_shots) # <<< USES noisy_sim
                    counts = qasm_job.result().get_counts()
                    sorted_counts = dict(sorted(counts.items()))

                    # ... (Histogram plotting logic - unchanged) ...
                    hist_fig = go.Figure(go.Bar(x=list(sorted_counts.keys()), y=list(sorted_counts.values()), marker_color='indianred'))
                    hist_fig.update_layout(title=dict(text=f"Results from {num_shots} shots (Noisy)", font_color='#333333'),
                                           xaxis_title="Outcome (Classical Bit String)", yaxis_title="Counts",
                                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(230, 230, 230, 0.2)', font_color='#333333')
                    st.plotly_chart(hist_fig, use_container_width=True)
                    # ... (Rest of histogram info logic - unchanged) ...
                    if counts:
                        most_likely_outcome = max(counts, key=counts.get)
                        st.subheader("Most Probable Outcome")
                        st.markdown(f"### `{most_likely_outcome}`")
                        with st.expander("Show Raw Counts"):
                            st.json(sorted_counts)
                        if qc.num_qubits > 0:
                            readout_order = "".join([f"q{i}" for i in range(qc.num_qubits - 1, -1, -1)])
                            st.info(f"💡 **How to Read the Output:** The bit string `{most_likely_outcome}` corresponds to the qubits in the order **`{readout_order}`**.")
                        if note_text:
                            st.markdown(note_text)
                else:
                    st.info("Circuit contains no classical registers for measurement outcomes.")

            # --- 2. Noisy Quantum State Analysis ---
            with st.spinner(f"Calculating noisy quantum state..."):
                st.header(f"Qubit State Analysis ({noise_type})")
                st.markdown("This analysis shows the *density matrix* of the system *before* measurement, including noise.")

                # Prepare a circuit to get the final density matrix
                qc_for_state = remove_final_measurements_if_any(qc)
                qc_for_state.save_density_matrix() # <<< KEY: Tell simulator to save this
                
                # Run the simulation
                state_job = noisy_sim.run(qc_for_state) # <<< USES noisy_sim
                final_density_matrix = state_job.result().get_density_matrix(qc_for_state)

                if qc.num_qubits > 0:
                    cols = st.columns(qc.num_qubits)
                    for i in range(qc.num_qubits):
                        with cols[i]:
                            # Get the reduced density matrix for just this qubit
                            # We trace out all *other* qubits from the final noisy state
                            rho = partial_trace(final_density_matrix, 
                                                [q for q in range(qc.num_qubits) if q != i]).data
                            
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
        
        # ####################################################################
        # # --- End of new logic split ---
        # ####################################################################

    except Exception as e:
        st.error(f"An error occurred while processing the QASM file: {e}")
        st.warning("Please ensure the QASM is valid and that your environment includes qiskit-aer (`pip install qiskit-aer`).")
else:
    st.info("Please select an example or upload a .qasm file using the sidebar to begin.")
