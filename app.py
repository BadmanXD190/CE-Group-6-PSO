import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_instance
from src.moabc import run_moabc

st.set_page_config(page_title="MOABC Group 6", layout="wide")
st.title("MOABC for Multi Objective Knapsack")

with st.expander("Algorithm overview and workflow", expanded=True):
    st.markdown(
        """
        **MOABC** adapts the Artificial Bee Colony algorithm for multi objective optimisation.
        
        Workflow used in this app:
        - Employed bees generate neighbours to improve food sources
        - Onlooker bees select good food sources and explore around them
        - Scout bees replace stagnated solutions after a trial limit
        - An external Pareto archive stores nondominated feasible solutions
        - Hypervolume is tracked to monitor convergence quality
        """
    )

DATASET_DIR = "dataset"

with st.sidebar:
    st.header("Instance selection")
    inst = st.selectbox("Instance", ["inst01", "inst02", "inst03", "inst04", "inst05"], index=2)

    st.header("Basic settings")
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=1, step=1)
    cycles = st.slider("Cycles", 50, 1000, 200, 50)

    st.header("Colony configuration")
    colony_size = st.slider("Colony size", 20, 200, 80, 10)
    food_sources = st.slider("Food sources", 10, 150, 40, 5)

    st.header("Search behaviour")
    flip_rate = st.slider("Flip rate", 0.005, 0.2, 0.03, 0.005)
    limit = st.slider("Scout limit", 5, 200, 40, 5)

    st.header("Archive control")
    archive_max = st.slider("Archive max size", 50, 500, 300, 50)

    st.header("Target hypervolume")
    target_hv = st.number_input("Target HV", min_value=0.0, max_value=1.0, value=0.41, step=0.01)

    run_btn = st.button("Run MOABC", type="primary")

# load data
try:
    items_df, meta = load_instance(DATASET_DIR, inst)
    cap_w1 = float(meta["capacity_w1"])
except Exception as e:
    st.error(f"Failed to load dataset. {e}")
    st.stop()

st.caption(f"Instance: {inst} | Items: {len(items_df)} | capacity_w1: {cap_w1}")

if run_btn:
    with st.spinner("Running MOABC..."):
        pareto_df, conv_df, summary = run_moabc(
            items_df=items_df,
            cap_w1=cap_w1,
            colony_size=int(colony_size),
            food_sources=int(food_sources),
            cycles=int(cycles),
            limit=int(limit),
            flip_rate=float(flip_rate),
            seed=int(seed),
            archive_max=int(archive_max),
            target_hv=float(target_hv)
        )
    st.session_state["pareto_df"] = pareto_df
    st.session_state["conv_df"] = conv_df
    st.session_state["summary"] = summary

if "summary" in st.session_state:
    summary = st.session_state["summary"]
    pareto_df = st.session_state["pareto_df"]
    conv_df = st.session_state["conv_df"]

    # header metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final hypervolume", f"{summary['final_hypervolume']:.6f}")
    m2.metric("Pareto size", f"{summary['pareto_size']}")
    m3.metric("Runtime (s)", f"{summary['runtime_seconds']:.2f}")
    m4.metric("Reached target cycle", "-" if summary["reached_target_cycle"] is None else str(summary["reached_target_cycle"]))

    tab1, tab2, tab3, tab4 = st.tabs(["Run output", "Convergence", "Archive preview", "About metrics"])

    with tab1:
        st.subheader("Pareto front")
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(pareto_df["w2"], pareto_df["value"])
        plt.xlabel("Secondary weight (w2)")
        plt.ylabel("Total value")
        plt.title(f"Pareto Front MOABC on {inst}")
        plt.grid(True)
        st.pyplot(fig, clear_figure=True)

    with tab2:
        st.subheader("Hypervolume convergence")
        fig = plt.figure(figsize=(7, 5))
        plt.plot(conv_df["cycle"], conv_df["hypervolume"])
        plt.xlabel("Cycle")
        plt.ylabel("Hypervolume")
        plt.title(f"Hypervolume Convergence MOABC on {inst}")
        plt.grid(True)
        st.pyplot(fig, clear_figure=True)

    with tab3:
        st.subheader("Top archive solutions preview")
        st.write("Showing best 50 solutions sorted by value.")
        preview = pareto_df.sort_values("value", ascending=False).head(50)
        st.dataframe(preview, use_container_width=True)

    with tab4:
        st.markdown(
            """
            **Hypervolume (HV)** measures both convergence and diversity.
            Higher HV indicates a better Pareto front under the chosen normalisation.
            
            **Pareto size** indicates how many nondominated feasible solutions are stored.
            
            **Reached target cycle** shows the first cycle where HV became equal or higher than the set target.
            """
        )

    st.subheader("Export outputs")
    st.download_button("Download Pareto CSV", pareto_df.to_csv(index=False), file_name=f"{inst}_moabc_pareto.csv")
    st.download_button("Download Convergence CSV", conv_df.to_csv(index=False), file_name=f"{inst}_moabc_convergence.csv")

else:
    st.info("Select instance and parameters, then click Run MOABC.")
