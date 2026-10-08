"""
app.py
------
Streamlit dashboard for the AI-Based Predictive Maintenance System.

Run:
    streamlit run app.py
"""

import os
import sqlite3
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Project imports ────────────────────────────────────────────────────────
from src.predict import predict, load_model, interpret_risk, RISK_THRESHOLDS

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "predictions.db")

# ──────────────────────────────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────────────────────────────

def init_db():
    """Create the predictions table if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            machine_type  TEXT,
            air_temp      REAL,
            process_temp  REAL,
            rot_speed     REAL,
            torque        REAL,
            tool_wear     REAL,
            prediction    INTEGER,
            probability   REAL,
            risk          TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(inputs: dict, result: dict):
    """Insert one prediction record into the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
            (timestamp, machine_type, air_temp, process_temp,
             rot_speed, torque, tool_wear, prediction, probability, risk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        inputs["machine_type"],
        inputs["air_temp"],
        inputs["process_temp"],
        inputs["rot_speed"],
        inputs["torque"],
        inputs["tool_wear"],
        result["prediction"],
        round(result["probability"], 4),
        result["risk"],
    ))
    conn.commit()
    conn.close()


def load_history(n: int = 10) -> pd.DataFrame:
    """Return the n most recent predictions from the database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {n}",
        conn
    )
    conn.close()
    return df


# ──────────────────────────────────────────────────────────────────────────
# Risk badge helper
# ──────────────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "Low Risk":    "#22c55e",   # green
    "Medium Risk": "#f59e0b",   # amber
    "High Risk":   "#ef4444",   # red
}

def risk_badge(risk: str) -> str:
    color = RISK_COLORS.get(risk, "#6b7280")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 12px;'
        f'border-radius:12px;font-weight:600;font-size:14px">{risk}</span>'
    )


# ──────────────────────────────────────────────────────────────────────────
# Gauge chart (pure Matplotlib — no external libs required)
# ──────────────────────────────────────────────────────────────────────────

def draw_gauge(probability: float) -> plt.Figure:
    """Draw a simple semicircular gauge for failure probability."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})

    # Background arc — full 180 °
    theta_full = np.linspace(0, np.pi, 200)
    ax.plot(theta_full, [1] * 200, color="#e5e7eb", linewidth=18, solid_capstyle="round")

    # Foreground arc — up to probability
    theta_prob = np.linspace(0, np.pi * probability, max(2, int(probability * 200)))
    color = (
        "#22c55e" if probability < RISK_THRESHOLDS["low"]
        else "#f59e0b" if probability < RISK_THRESHOLDS["medium"]
        else "#ef4444"
    )
    if len(theta_prob) > 1:
        ax.plot(theta_prob, [1] * len(theta_prob), color=color,
                linewidth=18, solid_capstyle="round")

    ax.set_ylim(0, 1.4)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.text(np.pi / 2, 0.15, f"{probability*100:.1f}%",
            ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.text(np.pi / 2, -0.35, "Failure Probability",
            ha="center", va="center", fontsize=9, color="#57606a")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Predictive Maintenance — Steel Plant",
        page_icon="⚙️",
        layout="wide",
    )

    init_db()

    # ── Load model once and cache it ───────────────────────────────────────
    @st.cache_resource
    def get_model():
        return load_model()

    model_loaded = False
    model = None
    try:
        model = get_model()
        model_loaded = True
    except FileNotFoundError as e:
        st.error(str(e))

    # ── Header ─────────────────────────────────────────────────────────────
    st.title("⚙️ AI-Based Predictive Maintenance System")
    st.markdown(
        "**Steel Plant Equipment Failure Prediction**  ·  "
        "UCI AI4I 2020 Dataset  ·  Random Forest Classifier"
    )
    st.markdown("---")

    # ── What is predictive maintenance? ───────────────────────────────────
    with st.expander("ℹ️ What is Predictive Maintenance?", expanded=False):
        st.markdown("""
**Predictive maintenance** uses machine sensor data to forecast equipment
failures *before* they happen — reducing unplanned downtime, maintenance
costs, and safety risks in industrial environments.

This prototype uses a **Random Forest** model trained on the
[UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

> ⚠️ This is a machine learning prototype for demonstration purposes.
> It is **not** a certified industrial safety system.  
> Risk categories are **model-based estimates**, not real safety thresholds.
        """)

    # ── Two-column layout ──────────────────────────────────────────────────
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("🔧 Sensor Input")

        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            help="L = Low quality, M = Medium quality, H = High quality",
        )

        air_temp = st.slider(
            "Air Temperature (K)",
            min_value=295.0, max_value=305.0, value=298.1, step=0.1,
            help="Typical range in dataset: 295–305 K",
        )
        process_temp = st.slider(
            "Process Temperature (K)",
            min_value=305.0, max_value=315.0, value=308.6, step=0.1,
            help="Typical range in dataset: 305–315 K",
        )
        rot_speed = st.slider(
            "Rotational Speed (rpm)",
            min_value=1168, max_value=2886, value=1551, step=1,
            help="Typical range in dataset: 1168–2886 rpm",
        )
        torque = st.slider(
            "Torque (Nm)",
            min_value=3.8, max_value=76.6, value=42.8, step=0.1,
            help="Typical range in dataset: 3.8–76.6 Nm",
        )
        tool_wear = st.slider(
            "Tool Wear (min)",
            min_value=0, max_value=253, value=0, step=1,
            help="Accumulated tool-wear time in minutes (0 = new tool)",
        )

        predict_btn = st.button("🔍 Predict Failure Risk", use_container_width=True)

    with col_result:
        st.subheader("📊 Prediction Result")

        if not model_loaded:
            st.warning("Model not loaded. Train the model first with `python -m src.train`.")

        elif predict_btn:
            inputs = dict(
                machine_type=machine_type,
                air_temp=air_temp,
                process_temp=process_temp,
                rot_speed=rot_speed,
                torque=torque,
                tool_wear=tool_wear,
            )

            with st.spinner("Running prediction …"):
                result = predict(**inputs, model=model)

            # ── Verdict ───────────────────────────────────────────────────
            if result["prediction"] == 1:
                st.error("🚨 **Equipment failure predicted!**")
            else:
                st.success("✅ **No failure predicted.**")

            st.markdown(
                f"**Risk Level:** {risk_badge(result['risk'])}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Failure Probability:** `{result['probability']*100:.1f}%`")

            # ── Gauge ─────────────────────────────────────────────────────
            fig = draw_gauge(result["probability"])
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            # ── Engineered feature preview ────────────────────────────────
            temp_diff = process_temp - air_temp
            power     = torque * rot_speed
            st.markdown("**Derived Feature Values:**")
            st.markdown(
                f"- `temp_diff` = {temp_diff:.2f} K  \n"
                f"- `power` = {power:,.0f} Nm·rpm"
            )

            # ── Save to history ───────────────────────────────────────────
            save_prediction(inputs, result)
            st.caption("✔ Prediction saved to history.")

        else:
            st.info("Adjust the sensor values on the left and click **Predict Failure Risk**.")

    # ── Prediction history ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗂️ Recent Prediction History")

    history = load_history(10)
    if history.empty:
        st.info("No predictions yet.  Run a prediction above to populate the history.")
    else:
        display_cols = [
            "timestamp", "machine_type",
            "air_temp", "process_temp", "rot_speed", "torque", "tool_wear",
            "prediction", "probability", "risk",
        ]
        rename_map = {
            "timestamp":    "Timestamp",
            "machine_type": "Type",
            "air_temp":     "Air Temp (K)",
            "process_temp": "Proc Temp (K)",
            "rot_speed":    "Speed (rpm)",
            "torque":       "Torque (Nm)",
            "tool_wear":    "Wear (min)",
            "prediction":   "Failure?",
            "probability":  "Prob.",
            "risk":         "Risk",
        }
        hist_display = history[display_cols].rename(columns=rename_map)
        hist_display["Failure?"] = hist_display["Failure?"].map({1: "Yes ⚠️", 0: "No ✅"})
        hist_display["Prob."]    = hist_display["Prob."].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(hist_display, use_container_width=True, hide_index=True)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "AI-Based Predictive Maintenance System · Built with Scikit-learn & Streamlit · "
        "Dataset: UCI AI4I 2020 · Model: Random Forest · "
        "⚠️ Prototype only — not for real industrial safety decisions."
    )


if __name__ == "__main__":
    main()
