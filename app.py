import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np

from pso.data_loader import list_instances, load_knapsack_instance
from pso.mopso_knapsack import mopso_knapsack

st.set_page_config(page_title="MOPSO Knapsack", layout="wide")

st.title("Multi Objective PSO for Knapsack")

st.write("Objective 1 maximize total value")
st.write("Objective 2 minimize total w2")
st.write("Constraint total w1 must be within capacity")

data_folder = st.sidebar.text_input("Processed instance folder", "processed_mknapcb3")

pairs = list_instances(data_folder)
if len(pairs) == 0:
    st.error("No instances found. Make sure the folder contains CSV and matching config JSON files.")
    st.stop()

options = [f"{i+1}. {pairs[i][0].split('/')[-1]}" for i in range(len(pairs))]
choice = st.sidebar.selectbox("Choose instance", options, index=0)
idx = options.index(choice)

csv_path, cfg_path = pairs[idx]
items, capacity_w1, cfg = load_knapsack_instance(csv_path, cfg_path)

st.sidebar.subheader("PSO parameters")
n_particles = st.sidebar.slider("Particles", 20, 200, 80, 10)
iters = st.sidebar.slider("Iterations", 50, 2000, 300, 50)
w = st.sidebar.slider("Inertia w", 0.1, 0.95, 0.7, 0.05)
c1 = st.sidebar.slider("c1", 0.5, 3.0, 1.6, 0.1)
c2 = st.sidebar.slider("c2", 0.5, 3.0, 1.6, 0.1)
archive_size = st.sidebar.slider("Archive size", 50, 500, 200, 25)
seed = st.sidebar.number_input("Seed", value=42, step=1)

run = st.sidebar.button("Run MOPSO")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Instance preview")
    st.write(f"CSV: {csv_path}")
    st.write(f"Capacity w1: {capacity_w1}")
    st.dataframe(items.head(10), use_container_width=True)

if run:
    with st.spinner("Running MOPSO..."):
        result = mopso_knapsack(
            items=items,
            capacity_w1=capacity_w1,
            n_particles=int(n_particles),
            iters=int(iters),
            w=float(w),
            c1=float(c1),
            c2=float(c2),
            archive_size=int(archive_size),
            seed=int(seed),
        )

    pareto_value = result["pareto_value"]
    pareto_w2 = result["pareto_w2"]

    with col2:
        st.subheader("Run summary")
        st.write(f"Runtime seconds: {result['runtime_s']:.3f}")
        st.write(f"Pareto solutions: {len(pareto_value)}")
        st.write(f"Best value found: {np.max(pareto_value):.2f}")
        st.write(f"Lowest w2 found: {np.min(pareto_w2):.2f}")

    st.subheader("Pareto front value vs w2")
    chart_df = pd.DataFrame({"value": pareto_value, "w2": pareto_w2})
    st.scatter_chart(chart_df, x="w2", y="value")

    st.subheader("Best value progress")
    hist_df = pd.DataFrame({"iter": np.arange(len(result["history_best_value"])), "best_value": result["history_best_value"]})
    st.line_chart(hist_df, x="iter", y="best_value")
else:
    with col2:
        st.info("Set parameters and click Run MOPSO")
