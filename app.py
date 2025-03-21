import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import time
from datetime import datetime
from functools import lru_cache
import sqlite3

# Must be the first Streamlit command
st.set_page_config(page_title="LLM Based B5G/6G Resource Allocator", layout="wide")

# Enhanced, Professional CSS with Subtle Animations
st.markdown("""
    <style>
    body {
        background-color: #1e2a38;
        color: #d0d7de;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #ff3366;
        color: #ffffff;
        border: 1px solid #ff6699;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(255, 51, 102, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b3b3;
        border-color: #00cccc;
        box-shadow: 0 4px 8px rgba(0, 179, 179, 0.5);
        transform: translateY(-2px);
    }
    .stop-button>button {
        background-color: #ff4d4d;
        border: 1px solid #ff8080;
        box-shadow: 0 2px 4px rgba(255, 77, 77, 0.3);
    }
    .stop-button>button:hover {
        background-color: #ff8080;
        border-color: #ff9999;
        box-shadow: 0 4px 8px rgba(255, 128, 128, 0.5);
        transform: translateY(-2px);
    }
    .stMetric {
        background: #2a3b4c;
        border: 1px solid #00b3b3;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0, 179, 179, 0.2);
        transition: box-shadow 0.3s ease;
    }
    .stMetric:hover {
        box-shadow: 0 2px 6px rgba(0, 179, 179, 0.4);
    }
    .stExpander {
        background: #2a3b4c;
        border: 1px solid #ff3366;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(255, 51, 102, 0.2);
    }
    .stExpander p {
        color: #d0d7de;
    }
    .ticker {
        background-color: #00b3b3;
        color: #ffffff;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0, 179, 179, 0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 5px rgba(0, 179, 179, 0.5); }
        50% { box-shadow: 0 0 10px rgba(0, 179, 179, 0.7); }
        100% { box-shadow: 0 0 5px rgba(0, 179, 179, 0.5); }
    }
    .sidebar .stSlider, .sidebar .stCheckbox, .sidebar .stSelectbox {
        background: #2a3b4c;
        padding: 8px;
        border-radius: 4px;
        transition: background 0.3s ease;
    }
    .sidebar .stSlider:hover, .sidebar .stCheckbox:hover, .sidebar .stSelectbox:hover {
        background: #334d66;
    }
    h1, h2 {
        color: #ffffff;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Database Setup
def init_db():
    conn = sqlite3.connect("allocation_history.db")
    c = conn.cursor()
    try:
        # Check if table exists and add missing columns if needed
        c.execute("PRAGMA table_info(allocations)")
        columns = [row[1] for row in c.fetchall()]
        if "available_bandwidth" not in columns or "available_power" not in columns or "raw_response" not in columns:
            if "available_bandwidth" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN available_bandwidth REAL DEFAULT 0")
            if "available_power" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN available_power REAL DEFAULT 0")
            if "raw_response" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN raw_response TEXT DEFAULT 'No AI summary'")
        else:
            # Create table with all columns if it doesn’t exist
            c.execute('''CREATE TABLE IF NOT EXISTS allocations 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          timestamp TEXT, 
                          data TEXT, 
                          energy_score INTEGER, 
                          health_score INTEGER, 
                          available_bandwidth REAL, 
                          available_power REAL, 
                          raw_response TEXT)''')
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {str(e)}")
    conn.commit()
    conn.close()

def save_allocation_to_db(allocation_df, energy_score, health_score, available_bandwidth, available_power, raw_response):
    conn = sqlite3.connect("allocation_history.db")
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    data_json = allocation_df.to_json()
    c.execute("INSERT INTO allocations (timestamp, data, energy_score, health_score, available_bandwidth, available_power, raw_response) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (timestamp, data_json, energy_score, health_score, available_bandwidth, available_power, raw_response))
    conn.commit()
    conn.close()

def get_allocation_history():
    conn = sqlite3.connect("allocation_history.db")
    c = conn.cursor()
    c.execute("SELECT id, timestamp FROM allocations ORDER BY timestamp DESC")
    history = [{"id": row[0], "timestamp": row[1]} for row in c.fetchall()]
    conn.close()
    return history

def load_allocation_by_id(allocation_id):
    conn = sqlite3.connect("allocation_history.db")
    c = conn.cursor()
    c.execute("SELECT data, energy_score, health_score, available_bandwidth, available_power, raw_response FROM allocations WHERE id = ?", (allocation_id,))
    result = c.fetchone()
    conn.close()
    if result:
        df = pd.read_json(result[0])
        return {
            "df": df,
            "energy_score": result[1] or 0,
            "health_score": result[2] or 0,
            "available_bandwidth": result[3] if result[3] is not None else 100,  # Default to 100 MHz if None
            "available_power": result[4] if result[4] is not None else 50,       # Default to 50 W if None
            "raw_response": result[5] or "No AI summary available"
        }
    return None

# API Configs
PERPLEXITY_API_KEY = "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"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

@lru_cache(maxsize=32)
def query_perplexity(user_ids_tuple, qos_requirements_tuple, available_bandwidth, available_power, ai_optimize=False):
    user_ids = list(user_ids_tuple)
    qos_requirements = json.loads(qos_requirements_tuple)
    
    prompt = f"""
    You are an AI expert in B5G and 6G wireless systems. Allocate bandwidth, power, and latency resources{' with AI optimization' if ai_optimize else ''}.

    Input:
    - User IDs: {user_ids}
    - QoS Requirements (JSON): {json.dumps(qos_requirements)}
    - Available Bandwidth (MHz): {available_bandwidth}
    - Available Power (Watts): {available_power}

    Output:
    Format: User_X: Y MHz, Z Watts, W ms latency
    Include: Energy efficiency score (0-100), Network health score (0-100)
    """

    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "system", "content": "Be precise and concise."}, {"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7 if not ai_optimize else 0.3,
    }

    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def parse_allocation_plan(response_text):
    if not response_text:
        st.warning("Empty API response.")
        return None, 0, 0

    lines = response_text.strip().split("\n")
    data = []
    energy_score = 0
    health_score = 0

    for line in lines:
        match = re.search(r"User_\d+:\s*(\d+)\s*MHz,\s*(\d+)\s*Watts,\s*(\d+)\s*ms", line, re.IGNORECASE)
        if match:
            bandwidth, power, latency = map(int, match.groups())
            user_id = line.split(":")[0].strip()
            data.append({"User ID": user_id, "Bandwidth (MHz)": bandwidth, "Power (Watts)": power, "Latency (ms)": latency, "QoS Requirement": "Custom"})
        score_match = re.search(r"(energy efficiency|network health) score:\s*(\d+)", line, re.IGNORECASE)
        if score_match:
            if "energy" in score_match.group(1).lower():
                energy_score = int(score_match.group(2))
            else:
                health_score = int(score_match.group(2))

    return pd.DataFrame(data) if data else None, energy_score, health_score

# Initialize Database
init_db()

# App Title
st.markdown("<h1 style='text-align: center;'>LLM Based B5G/6G Resource Allocator</h1>", unsafe_allow_html=True)

# Centered Controls - Sleek and Outstanding
st.markdown("<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
col_start, col_space, col_stop = st.columns([1, 2, 1])
with col_start:
    start_clicked = st.button("Start Allocation")
with col_stop:
    st.markdown('<div class="stop-button">', unsafe_allow_html=True)
    stop_clicked = st.button("Stop Allocation", key="stop_btn", help="Click to halt allocation")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
continuous_mode = st.checkbox("Continuous Mode", value=False, help="Enable to run allocation in a loop", key="continuous_mode")

# Sidebar - Configuration and History
st.sidebar.markdown("<h2 style='color: #00b3b3;'>Network Configuration</h2>", unsafe_allow_html=True)
num_users = st.sidebar.slider("Number of Users", 1, 20, 5)
available_bandwidth = st.sidebar.number_input("Available Bandwidth (MHz)", 10, 2000, 100)
available_power = st.sidebar.number_input("Available Power (Watts)", 10, 1000, 50)
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 60, 5)
ai_optimize = st.sidebar.checkbox("Enable AI Optimization", value=False)

user_profiles = []
for i in range(num_users):
    with st.sidebar.expander(f"User {i+1} Settings"):
        qos = st.selectbox(f"QoS Level", ["High", "Medium", "Low"], key=f"qos_{i}")
        latency = st.slider(f"Max Latency (ms)", 1, 100, 20, key=f"lat_{i}")
        reliability = st.slider(f"Reliability (%)", 50, 100, 95, key=f"rel_{i}")
        user_profiles.append({"qos": qos, "latency": latency, "reliability": reliability})

user_ids = [f"User_{i+1}" for i in range(num_users)]

# Sidebar - Latest Allocation and History Viewer
# Ensure allocation_data is initialized before accessing it
if "allocation_data" not in st.session_state:
    st.session_state.allocation_data = None

if st.session_state.allocation_data and not st.session_state.running:
    st.sidebar.markdown("<h2 style='color: #00b3b3;'>Allocation Options</h2>", unsafe_allow_html=True)
    if st.sidebar.button("View Latest Results", key="view_latest_sidebar"):
        data = st.session_state.allocation_data
        st.subheader("Latest Allocation Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bandwidth", f"{data['total_bw']}/{data['available_bandwidth']} MHz", f"{data['remaining_bw']} left")
        col2.metric("Power", f"{data['total_pw']}/{data['available_power']} W", f"{data['remaining_pw']} left")
        col3.metric("Latency", f"{data['avg_latency']:.1f} ms")
        col4.metric("Health Score", f"{data['health_score']}/100")
        st.progress(min(data['total_bw'] / data['available_bandwidth'], 1.0), "Bandwidth Usage")

        st.subheader("Resource Gauges")
        col5, col6 = st.columns(2)
        fig_gauge_bw = go.Figure(go.Indicator(mode="gauge+number", value=data['total_bw'], domain={'x': [0, 1], 'y': [0, 1]},
                                              title={'text': "Bandwidth (MHz)"}, gauge={'axis': {'range': [0, data['available_bandwidth']]}}))
        fig_gauge_pw = go.Figure(go.Indicator(mode="gauge+number", value=data['total_pw'], domain={'x': [0, 1], 'y': [0, 1]},
                                              title={'text': "Power (Watts)"}, gauge={'axis': {'range': [0, data['available_power']]}}))
        col5.plotly_chart(fig_gauge_bw, use_container_width=True, key="gauge_bw_static")
        col6.plotly_chart(fig_gauge_pw, use_container_width=True, key="gauge_pw_static")

        st.subheader("Allocation Overview")
        st.dataframe(data['df'].style.background_gradient(cmap="viridis"))

        st.subheader("Visual Insights")
        fig_bar = px.bar(data['df'], x="User ID", y=["Bandwidth (MHz)", "Power (Watts)", "Latency (ms)"], barmode="group")
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart_static")

        fig_3d = go.Figure(data=[go.Scatter3d(x=data['df']["Bandwidth (MHz)"], y=data['df']["Power (Watts)"],
                                              z=data['df']["Latency (ms)"], mode="markers",
                                              marker=dict(size=12, color=data['df'].index, colorscale="Blues"))])
        fig_3d.update_layout(title="6G Resource Space", scene=dict(xaxis_title="Bandwidth", yaxis_title="Power", zaxis_title="Latency"))
        st.plotly_chart(fig_3d, use_container_width=True, key="3d_chart_static")

        with st.expander("AI Summary", expanded=False):
            st.write(data['raw_response'] if data['raw_response'] else "No AI summary available.")

st.sidebar.markdown("<h2 style='color: #00b3b3;'>History Viewer</h2>", unsafe_allow_html=True)
show_history = st.sidebar.checkbox("Enable History Viewer", value=False, key="show_history")
if show_history:
    history = get_allocation_history()
    if history:
        selected_allocation = st.sidebar.selectbox("View Past Allocations", 
                                                   options=[f"{h['timestamp']} (ID: {h['id']})" for h in history],
                                                   format_func=lambda x: x)
        if selected_allocation:
            allocation_id = int(selected_allocation.split("ID: ")[1].strip(")"))
            past_data = load_allocation_by_id(allocation_id)
            if past_data:
                st.subheader(f"Allocation from {selected_allocation.split(' (')[0]}")
                col1, col2, col3, col4 = st.columns(4)
                total_bw = past_data['df']["Bandwidth (MHz)"].sum()
                total_pw = past_data['df']["Power (Watts)"].sum()
                avg_latency = past_data['df']["Latency (ms)"].mean()
                remaining_bw = past_data['available_bandwidth'] - total_bw if past_data['available_bandwidth'] is not None else 0
                remaining_pw = past_data['available_power'] - total_pw if past_data['available_power'] is not None else 0
                col1.metric("Bandwidth", f"{total_bw}/{past_data['available_bandwidth'] if past_data['available_bandwidth'] is not None else 100} MHz", f"{remaining_bw} left")
                col2.metric("Power", f"{total_pw}/{past_data['available_power'] if past_data['available_power'] is not None else 50} W", f"{remaining_pw} left")
                col3.metric("Latency", f"{avg_latency:.1f} ms")
                col4.metric("Health Score", f"{past_data['health_score']}/100")
                st.progress(min(total_bw / (past_data['available_bandwidth'] if past_data['available_bandwidth'] is not None else 100), 1.0), "Bandwidth Usage")

                st.subheader("Resource Gauges")
                col5, col6 = st.columns(2)
                fig_gauge_bw = go.Figure(go.Indicator(mode="gauge+number", value=total_bw, domain={'x': [0, 1], 'y': [0, 1]},
                                                      title={'text': "Bandwidth (MHz)"}, gauge={'axis': {'range': [0, past_data['available_bandwidth'] if past_data['available_bandwidth'] is not None else 100]}}))
                fig_gauge_pw = go.Figure(go.Indicator(mode="gauge+number", value=total_pw, domain={'x': [0, 1], 'y': [0, 1]},
                                                      title={'text': "Power (Watts)"}, gauge={'axis': {'range': [0, past_data['available_power'] if past_data['available_power'] is not None else 50]}}))
                col5.plotly_chart(fig_gauge_bw, use_container_width=True, key=f"gauge_bw_hist_{allocation_id}")
                col6.plotly_chart(fig_gauge_pw, use_container_width=True, key=f"gauge_pw_hist_{allocation_id}")

                st.subheader("Allocation Overview")
                st.dataframe(past_data['df'].style.background_gradient(cmap="viridis"))

                st.subheader("Visual Insights")
                fig_bar = px.bar(past_data['df'], x="User ID", y=["Bandwidth (MHz)", "Power (Watts)", "Latency (ms)"], barmode="group")
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_hist_{allocation_id}")

                fig_3d = go.Figure(data=[go.Scatter3d(x=past_data['df']["Bandwidth (MHz)"], y=past_data['df']["Power (Watts)"],
                                                      z=past_data['df']["Latency (ms)"], mode="markers",
                                                      marker=dict(size=12, color=past_data['df'].index, colorscale="Blues"))])
                fig_3d.update_layout(title="6G Resource Space", scene=dict(xaxis_title="Bandwidth", yaxis_title="Power", zaxis_title="Latency"))
                st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_chart_hist_{allocation_id}")

                st.download_button("Export This Allocation", json.dumps({"allocation": past_data['df'].to_dict(), "scores": {"energy": past_data['energy_score'], "health": past_data['health_score']}, "resources": {"bandwidth": past_data['available_bandwidth'] if past_data['available_bandwidth'] is not None else 100, "power": past_data['available_power'] if past_data['available_power'] is not None else 50}}), f"allocation_{allocation_id}.json", "application/json", key=f"download_hist_{allocation_id}")

                with st.expander("AI Summary", expanded=False):
                    st.write(past_data['raw_response'] if past_data['raw_response'] else "No AI summary available.")

# Initialize Session State
if "running" not in st.session_state:
    st.session_state.running = False
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "allocation_data" not in st.session_state:
    st.session_state.allocation_data = None  # Explicitly initialize to None

# Main Dashboard
if start_clicked and not st.session_state.running:
    st.session_state.running = True
    st.session_state.iteration = 0
    st.session_state.allocation_data = None

if st.session_state.running:
    with st.spinner("Allocating Resources..."):
        while st.session_state.running:
            st.session_state.iteration += 1
            iteration_key = f"iter_{st.session_state.iteration}"
            
            qos_json = json.dumps(user_profiles)
            response = query_perplexity(tuple(user_ids), qos_json, available_bandwidth, available_power, ai_optimize)
            
            with st.expander("AI Summary", expanded=False):
                st.write(response if response else "No AI summary received.")

            allocation_df, energy_score, health_score = parse_allocation_plan(response)

            if allocation_df is not None:
                for i, row in allocation_df.iterrows():
                    allocation_df.at[i, "QoS Requirement"] = user_profiles[i]["qos"]

                total_bw = allocation_df["Bandwidth (MHz)"].sum()
                total_pw = allocation_df["Power (Watts)"].sum()
                avg_latency = allocation_df["Latency (ms)"].mean()
                remaining_bw = available_bandwidth - total_bw
                remaining_pw = available_power - total_pw

                st.session_state.allocation_data = {
                    "df": allocation_df,
                    "energy_score": energy_score,
                    "health_score": health_score,
                    "total_bw": total_bw,
                    "total_pw": total_pw,
                    "avg_latency": avg_latency,
                    "remaining_bw": remaining_bw,
                    "remaining_pw": remaining_pw,
                    "available_bandwidth": available_bandwidth,
                    "available_power": available_power,
                    "raw_response": response
                }

                save_allocation_to_db(allocation_df, energy_score, health_score, available_bandwidth, available_power, response)

                # Display results during allocation
                data = st.session_state.allocation_data
                st.subheader("Allocation Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Bandwidth", f"{data['total_bw']}/{data['available_bandwidth']} MHz", f"{data['remaining_bw']} left")
                col2.metric("Power", f"{data['total_pw']}/{data['available_power']} W", f"{data['remaining_pw']} left")
                col3.metric("Latency", f"{data['avg_latency']:.1f} ms")
                col4.metric("Health Score", f"{data['health_score']}/100")
                st.progress(min(data['total_bw'] / data['available_bandwidth'], 1.0), "Bandwidth Usage")

                st.subheader("Resource Gauges")
                col5, col6 = st.columns(2)
                fig_gauge_bw = go.Figure(go.Indicator(mode="gauge+number", value=data['total_bw'], domain={'x': [0, 1], 'y': [0, 1]},
                                                      title={'text': "Bandwidth (MHz)"}, gauge={'axis': {'range': [0, data['available_bandwidth']]}}))
                fig_gauge_pw = go.Figure(go.Indicator(mode="gauge+number", value=data['total_pw'], domain={'x': [0, 1], 'y': [0, 1]},
                                                      title={'text': "Power (Watts)"}, gauge={'axis': {'range': [0, data['available_power']]}}))
                col5.plotly_chart(fig_gauge_bw, use_container_width=True, key=f"gauge_bw_{iteration_key}")
                col6.plotly_chart(fig_gauge_pw, use_container_width=True, key=f"gauge_pw_{iteration_key}")

                st.subheader("Allocation Overview")
                st.dataframe(data['df'].style.background_gradient(cmap="viridis"))

                st.subheader("Visual Insights")
                fig_bar = px.bar(data['df'], x="User ID", y=["Bandwidth (MHz)", "Power (Watts)", "Latency (ms)"], barmode="group")
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{iteration_key}")

                fig_3d = go.Figure(data=[go.Scatter3d(x=data['df']["Bandwidth (MHz)"], y=data['df']["Power (Watts)"],
                                                      z=data['df']["Latency (ms)"], mode="markers",
                                                      marker=dict(size=12, color=data['df'].index, colorscale="Blues"))])
                fig_3d.update_layout(title="6G Resource Space", scene=dict(xaxis_title="Bandwidth", yaxis_title="Power", zaxis_title="Latency"))
                st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_chart_{iteration_key}")

                if data['total_bw'] > data['available_bandwidth'] or data['total_pw'] > data['available_power']:
                    st.error("Over-allocation detected!")
                if any(data['df']["Latency (ms)"] > [p["latency"] for p in user_profiles]):
                    st.warning("Latency exceeds QoS requirements!")

                st.download_button("Export Current Allocation", json.dumps({"allocation": data['df'].to_dict(), "scores": {"energy": energy_score, "health": energy_score}, "resources": {"bandwidth": data['available_bandwidth'], "power": data['available_power']}}), "allocation.json", "application/json", key=f"download_{iteration_key}")

                st.markdown(f"<span class='ticker'>Status: Active - Iteration {st.session_state.iteration}</span>", unsafe_allow_html=True)

            if not continuous_mode:
                st.session_state.running = False
            if stop_clicked:
                st.session_state.running = False
                break
            if st.session_state.running and continuous_mode:
                time.sleep(refresh_rate)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #00b3b3; font-size: 14px;'>Developed by KK | Powered by 6G Innovation</p>", unsafe_allow_html=True)