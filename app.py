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
import io
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Must be the first Streamlit command
st.set_page_config(page_title="LLM Based B5G/6G Resource Allocator", layout="wide")

# Enhanced CSS with Vibrant Colors, Borders, and Highlights
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #333333;
        margin: 0;
        padding: 0;
    }
    .stButton>button {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .stop-button>button {
        background-color: #ff6f61;
        border: none;
    }
    .stop-button>button:hover {
        background-color: #e65a50;
        transform: translateY(-2px);
    }
    .stMetric {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        background: linear-gradient(145deg, #ffffff, #f1f3f5);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .stExpander {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stExpander p {
        color: #333333;
        font-size: 14px;
    }
    .sidebar .stSlider, .sidebar .stCheckbox, .sidebar .stSelectbox {
        background: #ffffff;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        transition: background 0.3s ease;
    }
    .sidebar .stSlider:hover, .sidebar .stCheckbox:hover, .sidebar .stSelectbox:hover {
        background: #f1f3f5;
    }
    h1, h2, h3 {
        color: #007bff;
        font-weight: 600;
    }
    h1 {
        font-size: 36px;
        margin-bottom: 10px;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    h2 {
        font-size: 26px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    h3 {
        font-size: 20px;
        margin-top: 15px;
        margin-bottom: 8px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stPlotlyChart {
        background: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    .stPlotlyChart:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background-color: #e9ecef;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 14px;
        color: #555555;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .quick-stats {
        background: #ffffff;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-around;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 180px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -90px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .stProgress .st-bo {
        background-color: #00c4b4;
    }
    .stSuccess {
        color: #00c4b4;
        font-weight: 600;
        background: #e6f7fa;
        padding: 8px;
        border-radius: 6px;
        border-left: 4px solid #00c4b4;
    }
    .stWarning {
        color: #ff6f61;
        font-weight: 600;
        background: #ffe6e6;
        padding: 8px;
        border-radius: 6px;
        border-left: 4px solid #ff6f61;
    }
    .highlight-insight {
        background: #fff3cd;
        padding: 10px;
        border-radius: 6px;
        border-left: 4px solid #ffca2c;
        font-weight: 500;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Database Setup
def init_db():
    conn = sqlite3.connect("allocation_history.db")
    c = conn.cursor()
    try:
        c.execute("PRAGMA table_info(allocations)")
        columns = [row[1] for row in c.fetchall()]
        if "available_bandwidth" not in columns or "available_power" not in columns or "raw_response" not in columns or "metrics" not in columns:
            if "available_bandwidth" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN available_bandwidth REAL DEFAULT 0")
            if "available_power" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN available_power REAL DEFAULT 0")
            if "raw_response" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN raw_response TEXT DEFAULT 'No AI summary'")
            if "metrics" not in columns:
                c.execute("ALTER TABLE allocations ADD COLUMN metrics TEXT DEFAULT '{}'")
        else:
            c.execute('''CREATE TABLE IF NOT EXISTS allocations 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          timestamp TEXT, 
                          data TEXT, 
                          energy_score INTEGER, 
                          health_score INTEGER, 
                          available_bandwidth REAL, 
                          available_power REAL, 
                          raw_response TEXT,
                          metrics TEXT)''')
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {str(e)}")
    conn.commit()
    conn.close()

def save_allocation_to_db(allocation_df, energy_score, health_score, available_bandwidth, available_power, raw_response, metrics):
    metrics_converted = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v for k, v in metrics.items()}
    conn = sqlite3.connect("allocation_history.db")
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    data_json = allocation_df.to_json()
    metrics_json = json.dumps(metrics_converted)
    c.execute("INSERT INTO allocations (timestamp, data, energy_score, health_score, available_bandwidth, available_power, raw_response, metrics) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (timestamp, data_json, energy_score, health_score, available_bandwidth, available_power, raw_response, metrics_json))
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
    c.execute("SELECT data, energy_score, health_score, available_bandwidth, available_power, raw_response, metrics FROM allocations WHERE id = ?", (allocation_id,))
    result = c.fetchone()
    conn.close()
    if result:
        df = pd.read_json(result[0])
        metrics = json.loads(result[6]) if result[6] else {}
        return {
            "df": df,
            "energy_score": result[1] or 0,
            "health_score": result[2] or 0,
            "available_bandwidth": result[3] if result[3] is not None else 100,
            "available_power": result[4] if result[4] is not None else 50,
            "raw_response": result[5] or "No AI summary available",
            "metrics": metrics
        }
    return None

# API Configs for Gemini
GEMINI_API_KEY = "AIzaSyB2XZ3YVeaP-KyU6uG5fp5hGPSrXFqYXOE"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

@lru_cache(maxsize=32)
def query_gemini(user_ids_tuple, qos_requirements_tuple, available_bandwidth, available_power, ai_optimize=False):
    user_ids = list(user_ids_tuple)
    qos_requirements = json.loads(qos_requirements_tuple)
    
    prompt = f"""
    You are an AI expert in B5G and 6G wireless systems. Your task is to allocate bandwidth, power, and latency resources{' with AI optimization' if ai_optimize else ''} for the given users based on their QoS requirements. Provide a detailed explanation of the allocation process, including why resources were allocated in this way, how QoS requirements were met, and any trade-offs made. Do NOT provide a code snippet or implementation details.

    Input:
    - User IDs: {user_ids}
    - QoS Requirements (JSON): {json.dumps(qos_requirements)}
    - Available Bandwidth (MHz): {available_bandwidth}
    - Available Power (Watts): {available_power}

    Output:
    First, provide the allocation for ALL users specified in the User IDs list, even if resources need to be distributed unequally. Ensure the number of allocations matches the number of users ({len(user_ids)}). Format each allocation as follows, one per line:
    User_X: Y MHz, Z Watts, W ms latency
    After the allocations, include the scores on separate lines in the following format:
    Energy efficiency score: S
    Network health score: T
    Where Y, Z, W, S, and T are integers. Ensure the total bandwidth and power do not exceed the available amounts. If QoS requirements cannot be met, allocate resources as fairly as possible.

    Then, provide an explanatory summary in the following format:
    Explanation:
    - Describe how the resources were allocated to each user.
    - Explain how the QoS requirements (e.g., latency, reliability) influenced the allocation.
    - Highlight any trade-offs made (e.g., prioritizing latency over bandwidth for certain users).
    - Discuss the energy efficiency and network health scores, including factors that contributed to these scores.

    Example Output for 5 users:
    User_1: 20 MHz, 10 Watts, 20 ms latency
    User_2: 20 MHz, 10 Watts, 20 ms latency
    User_3: 20 MHz, 10 Watts, 20 ms latency
    User_4: 20 MHz, 10 Watts, 20 ms latency
    User_5: 20 MHz, 10 Watts, 20 ms latency
    Energy efficiency score: 90
    Network health score: 95

    Explanation:
    - Resources were allocated equally among users to ensure fairness, with each user receiving 20 MHz and 10 Watts.
    - Latency was set to 20 ms to meet the High QoS requirement for all users, ensuring low-latency communication.
    - No trade-offs were necessary as the available bandwidth (100 MHz) and power (50 Watts) were sufficient.
    - The energy efficiency score of 90 reflects optimal power usage, while the network health score of 95 indicates high QoS satisfaction.
    """

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7 if not ai_optimize else 0.3,
            "maxOutputTokens": 2048
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        response_json = response.json()
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.error("No valid response from Gemini API. Response: " + str(response_json))
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}. Response: {e.response.text if e.response else 'No response'}")
        return None

def parse_allocation_plan(response_text, expected_users, available_bandwidth, available_power):
    if not response_text:
        st.warning("Empty API response. Using default allocation.")
        return generate_default_allocation(expected_users, available_bandwidth, available_power), 50, 50

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
        else:
            partial_match = re.search(r"User_\d+:\s*(\d+)\s*MHz,\s*(\d+)\s*Watts,\s*(\d+)", line, re.IGNORECASE)
            if partial_match:
                bandwidth, power, latency = map(int, partial_match.groups())
                user_id = line.split(":")[0].strip()
                data.append({"User ID": user_id, "Bandwidth (MHz)": bandwidth, "Power (Watts)": power, "Latency (ms)": latency, "QoS Requirement": "Custom"})
        
        score_match = re.search(r"(energy efficiency|network health) score:\s*(\d+)", line, re.IGNORECASE)
        if score_match:
            if "energy" in score_match.group(1).lower():
                energy_score = int(score_match.group(2))
            else:
                health_score = int(score_match.group(2))

    if len(data) < expected_users:
        st.warning(f"Expected allocations for {expected_users} users, but only parsed {len(data)} users. Generating default allocations for missing users. Raw response:\n{response_text}")
        parsed_user_ids = {entry["User ID"] for entry in data}
        for i in range(1, expected_users + 1):
            user_id = f"User_{i}"
            if user_id not in parsed_user_ids:
                default_bandwidth = available_bandwidth // expected_users if expected_users > 0 else 0
                default_power = available_power // expected_users if expected_users > 0 else 0
                default_latency = 20
                data.append({
                    "User ID": user_id,
                    "Bandwidth (MHz)": default_bandwidth,
                    "Power (Watts)": default_power,
                    "Latency (ms)": default_latency,
                    "QoS Requirement": "Custom"
                })
        data.sort(key=lambda x: int(x["User ID"].split("_")[1]))

    if energy_score == 0:
        st.warning("Energy Efficiency Score not found in response. Defaulting to 50.")
        energy_score = 50
    if health_score == 0:
        st.warning("Network Health Score not found in response. Defaulting to 50.")
        health_score = 50

    # Validate allocation
    total_bandwidth = sum(entry["Bandwidth (MHz)"] for entry in data)
    total_power = sum(entry["Power (Watts)"] for entry in data)
    if total_bandwidth > available_bandwidth:
        st.error(f"Over-allocation of bandwidth: {total_bandwidth} MHz allocated, but only {available_bandwidth} MHz available.")
        # Scale down bandwidth proportionally
        scale_factor = available_bandwidth / total_bandwidth if total_bandwidth > 0 else 1
        for entry in data:
            entry["Bandwidth (MHz)"] = int(entry["Bandwidth (MHz)"] * scale_factor)
    if total_power > available_power:
        st.error(f"Over-allocation of power: {total_power} Watts allocated, but only {available_power} Watts available.")
        # Scale down power proportionally
        scale_factor = available_power / total_power if total_power > 0 else 1
        for entry in data:
            entry["Power (Watts)"] = int(entry["Power (Watts)"] * scale_factor)

    return pd.DataFrame(data), energy_score, health_score

def generate_default_allocation(num_users, available_bandwidth, available_power):
    data = []
    for i in range(1, num_users + 1):
        user_id = f"User_{i}"
        default_bandwidth = available_bandwidth // num_users if num_users > 0 else 0
        default_power = available_power // num_users if num_users > 0 else 0
        default_latency = 20
        data.append({
            "User ID": user_id,
            "Bandwidth (MHz)": default_bandwidth,
            "Power (Watts)": default_power,
            "Latency (ms)": default_latency,
            "QoS Requirement": "Custom"
        })
    return pd.DataFrame(data)

# Function to Generate PDF Report
def generate_pdf_report(allocation_df, metrics, energy_score, health_score, raw_response):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Cover Page
    story.append(Paragraph("B5G/6G Resource Allocation Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 300))
    story.append(Paragraph("Developed by KK | Powered by 6G Innovation", styles['Normal']))
    story.append(Spacer(1, 12))
    doc.build(story)
    story = []  # Reset story for the next page

    # Allocation Details
    story.append(Paragraph("Allocation Details", styles['Heading2']))
    for _, row in allocation_df.iterrows():
        story.append(Paragraph(f"{row['User ID']}: {row['Bandwidth (MHz)']} MHz, {row['Power (Watts)']} Watts, {row['Latency (ms)']} ms latency", styles['Normal']))
        story.append(Spacer(1, 6))

    # Metrics
    story.append(Paragraph("Metrics", styles['Heading2']))
    story.append(Paragraph(f"Total Bandwidth: {metrics['total_bw']}/{metrics['available_bandwidth']} MHz", styles['Normal']))
    story.append(Paragraph(f"Total Power: {metrics['total_pw']}/{metrics['available_power']} W", styles['Normal']))
    story.append(Paragraph(f"Average Latency: {metrics['avg_latency']:.1f} ms", styles['Normal']))
    story.append(Paragraph(f"Network Health: {metrics['health_score']}/100", styles['Normal']))
    story.append(Paragraph(f"Bandwidth Utilization: {metrics['bw_utilization']:.1f}%", styles['Normal']))
    story.append(Paragraph(f"Power Utilization: {metrics['pw_utilization']:.1f}%", styles['Normal']))
    story.append(Paragraph(f"Energy Efficiency: {energy_score}/100", styles['Normal']))
    story.append(Spacer(1, 12))

    # AI Summary
    story.append(Paragraph("AI Summary", styles['Heading2']))
    if raw_response:
        for line in raw_response.split("\n"):
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No AI summary available.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Function to Perform Allocation
def perform_allocation(user_ids, user_profiles, available_bandwidth, available_power, ai_optimize):
    qos_json = json.dumps(user_profiles)
    response = query_gemini(tuple(user_ids), qos_json, available_bandwidth, available_power, ai_optimize)
    
    if not response:
        st.error("Failed to get allocation from AI. Using default allocation.")
        allocation_df = generate_default_allocation(len(user_ids), available_bandwidth, available_power)
        energy_score = 50
        health_score = 50
    else:
        allocation_df, energy_score, health_score = parse_allocation_plan(response, expected_users=len(user_ids), available_bandwidth=available_bandwidth, available_power=available_power)

    # Add QoS requirements to the DataFrame
    for i, row in allocation_df.iterrows():
        user_index = int(row["User ID"].split("_")[1]) - 1
        if user_index < len(user_profiles):
            allocation_df.at[i, "QoS Requirement"] = user_profiles[user_index]["qos"]

    total_bw = allocation_df["Bandwidth (MHz)"].sum()
    total_pw = allocation_df["Power (Watts)"].sum()
    avg_latency = allocation_df["Latency (ms)"].mean()
    remaining_bw = available_bandwidth - total_bw
    remaining_pw = available_power - total_pw

    # Calculate utilization metrics
    bw_utilization = (total_bw / available_bandwidth) * 100 if available_bandwidth > 0 else 0
    pw_utilization = (total_pw / available_power) * 100 if available_power > 0 else 0

    metrics = {
        "total_bw": total_bw,
        "total_pw": total_pw,
        "avg_latency": avg_latency,
        "remaining_bw": remaining_bw,
        "remaining_pw": remaining_pw,
        "available_bandwidth": available_bandwidth,
        "available_power": available_power,
        "bw_utilization": bw_utilization,
        "pw_utilization": pw_utilization,
        "health_score": health_score
    }

    return {
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
        "raw_response": response,
        "metrics": metrics
    }

# Initialize Database
init_db()

# Session State for Navigation and Allocation
if "display_history" not in st.session_state:
    st.session_state.display_history = False
if "display_help" not in st.session_state:
    st.session_state.display_help = False
if "running" not in st.session_state:
    st.session_state.running = False
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "allocation_data" not in st.session_state:
    st.session_state.allocation_data = None

# App Title and Introduction
st.markdown("<h1>LLM-Based B5G/6G Resource Allocator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; font-size: 14px; margin-bottom: 15px;'>Optimize resource allocation for Beyond 5G and 6G networks with AI-driven insights.</p>", unsafe_allow_html=True)

# Quick Stats Widget
st.markdown('<div class="quick-stats">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.markdown('<div class="tooltip">Total Users<span class="tooltiptext">Number of users currently configured for allocation.</span></div>', unsafe_allow_html=True)
col1.metric("", f"{st.session_state.get('num_users', 5)}", delta=None)
col2.markdown('<div class="tooltip">Available Bandwidth<span class="tooltiptext">Total bandwidth available for allocation (MHz).</span></div>', unsafe_allow_html=True)
col2.metric("", f"{st.session_state.get('available_bandwidth', 100)} MHz", delta=None)
col3.markdown('<div class="tooltip">Available Power<span class="tooltiptext">Total power available for allocation (Watts).</span></div>', unsafe_allow_html=True)
col3.metric("", f"{st.session_state.get('available_power', 50)} W", delta=None)
st.markdown('</div>', unsafe_allow_html=True)

# Centered Controls
st.markdown("<div style='text-align: center; margin: 15px 0;'>", unsafe_allow_html=True)
col_start, col_space, col_stop = st.columns([1, 2, 1])
with col_start:
    start_clicked = st.button("Start Allocation")
with col_stop:
    st.markdown('<div class="stop-button">', unsafe_allow_html=True)
    stop_clicked = st.button("Stop Allocation", key="stop_btn", help="Click to halt allocation")
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
continuous_mode = st.checkbox("Continuous Mode", value=False, help="Enable to run allocation in a loop", key="continuous_mode")

# Sidebar - Configuration
st.sidebar.markdown("<h2 style='color: #007bff;'>Network Configuration</h2>", unsafe_allow_html=True)
num_users = st.sidebar.slider("Number of Users", 1, 20, 5)
st.session_state.num_users = num_users
available_bandwidth = st.sidebar.number_input("Available Bandwidth (MHz)", 10, 2000, 100)
st.session_state.available_bandwidth = available_bandwidth
available_power = st.sidebar.number_input("Available Power (Watts)", 10, 1000, 50)
st.session_state.available_power = available_power
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

# Sidebar - Latest Allocation
if st.session_state.allocation_data and not st.session_state.running:
    st.sidebar.markdown("<h2 style='color: #007bff;'>Allocation Options</h2>", unsafe_allow_html=True)
    if st.sidebar.button("View Latest Results", key="view_latest_sidebar"):
        data = st.session_state.allocation_data
        st.subheader("Latest Allocation Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown('<div class="tooltip">Total Bandwidth<span class="tooltiptext">Allocated/Available Bandwidth (MHz).</span></div>', unsafe_allow_html=True)
        col1.metric("", f"{data['total_bw']}/{data['available_bandwidth']} MHz", f"{data['remaining_bw']} left")
        col2.markdown('<div class="tooltip">Total Power<span class="tooltiptext">Allocated/Available Power (Watts).</span></div>', unsafe_allow_html=True)
        col2.metric("", f"{data['total_pw']}/{data['available_power']} W", f"{data['remaining_pw']} left")
        col3.markdown('<div class="tooltip">Average Latency<span class="tooltiptext">Average latency across all users (ms).</span></div>', unsafe_allow_html=True)
        col3.metric("", f"{data['avg_latency']:.1f} ms")
        col4.markdown('<div class="tooltip">Network Health<span class="tooltiptext">Overall network health score (0-100).</span></div>', unsafe_allow_html=True)
        col4.metric("", f"{data['health_score']}/100")
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
        fig = px.bar(data['df'], x="User ID", y=["Bandwidth (MHz)", "Power (Watts)", "Latency (ms)"], barmode="group", title="Resource Allocation per User")
        st.plotly_chart(fig, use_container_width=True, key="chart_static_latest")

        fig_3d = go.Figure(data=[go.Scatter3d(x=data['df']["Bandwidth (MHz)"], y=data['df']["Power (Watts)"],
                                              z=data['df']["Latency (ms)"], mode="markers",
                                              marker=dict(size=8, color=data['df'].index, colorscale="Blues"))])
        fig_3d.update_layout(title="6G Resource Space", scene=dict(xaxis_title="Bandwidth", yaxis_title="Power", zaxis_title="Latency"))
        st.plotly_chart(fig_3d, use_container_width=True, key="3d_chart_static_latest")

        with st.expander("AI Summary", expanded=False):
            st.write(data['raw_response'] if data['raw_response'] else "No AI summary available.")

# Sidebar - Toggle Buttons
st.sidebar.markdown("<h2 style='color: #007bff;'>View Options</h2>", unsafe_allow_html=True)
if st.sidebar.button("Toggle History View", key="toggle_history"):
    st.session_state.display_history = not st.session_state.display_history
    st.session_state.display_help = False
if st.sidebar.button("Know Your Dashboard", key="toggle_help"):
    st.session_state.display_help = not st.session_state.display_help
    st.session_state.display_history = False

# History View
if st.session_state.display_history:
    history = get_allocation_history()
    if history:
        selected_allocation = st.selectbox("View Past Allocations", 
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
                fig = px.bar(past_data['df'], x="User ID", y=["Bandwidth (MHz)", "Power (Watts)", "Latency (ms)"], barmode="group", title="Resource Allocation per User")
                st.plotly_chart(fig, use_container_width=True, key=f"chart_hist_{allocation_id}")

                fig_3d = go.Figure(data=[go.Scatter3d(x=past_data['df']["Bandwidth (MHz)"], y=past_data['df']["Power (Watts)"],
                                                      z=past_data['df']["Latency (ms)"], mode="markers",
                                                      marker=dict(size=8, color=past_data['df'].index, colorscale="Blues"))])
                fig_3d.update_layout(title="6G Resource Space", scene=dict(xaxis_title="Bandwidth", yaxis_title="Power", zaxis_title="Latency"))
                st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_chart_hist_{allocation_id}")

                # Display Saved Metrics
                st.subheader("Saved Metrics")
                metrics = past_data['metrics']
                col1, col2, col3 = st.columns(3)
                col1.metric("Bandwidth Utilization", f"{metrics.get('bw_utilization', 0):.1f}%")
                col2.metric("Power Utilization", f"{metrics.get('pw_utilization', 0):.1f}%")
                col3.metric("Energy Efficiency", f"{past_data['energy_score']}/100")

                with st.expander("AI Summary", expanded=False):
                    st.write(past_data['raw_response'] if past_data['raw_response'] else "No AI summary available.")

# Default values for dashboard explanation
default_allocation_df = pd.DataFrame(columns=["User ID", "Bandwidth (MHz)", "Power (Watts)", "Latency (ms)", "QoS Requirement"])
default_bw_utilization = 0
default_energy_score = 0

# Allocation Logic
if start_clicked and not st.session_state.running:
    st.session_state.running = True
    st.session_state.iteration = 0
    st.session_state.allocation_data = None
    st.session_state.display_history = False
    st.session_state.display_help = False

if st.session_state.running:
    with st.spinner("Allocating Resources..."):
        while st.session_state.running:
            st.session_state.iteration += 1
            iteration_key = f"iter_{st.session_state.iteration}"
            
            # Perform allocation
            allocation_data = perform_allocation(user_ids, user_profiles, available_bandwidth, available_power, ai_optimize)
            st.session_state.allocation_data = allocation_data

            # Save to database
            save_allocation_to_db(
                allocation_data["df"],
                allocation_data["energy_score"],
                allocation_data["health_score"],
                allocation_data["available_bandwidth"],
                allocation_data["available_power"],
                allocation_data["raw_response"],
                allocation_data["metrics"]
            )

            # Display Results
            data = st.session_state.allocation_data
            with st.expander("AI Summary", expanded=True):
                st.markdown("### AI Summary")
                st.write(data['raw_response'] if data['raw_response'] else "No AI summary received.")

            st.subheader("Allocation Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown('<div class="tooltip">Total Bandwidth<span class="tooltiptext">Allocated/Available Bandwidth (MHz).</span></div>', unsafe_allow_html=True)
            col1.metric("", f"{data['total_bw']}/{data['available_bandwidth']} MHz", f"{data['remaining_bw']} left")
            col2.markdown('<div class="tooltip">Total Power<span class="tooltiptext">Allocated/Available Power (Watts).</span></div>', unsafe_allow_html=True)
            col2.metric("", f"{data['total_pw']}/{data['available_power']} W", f"{data['remaining_pw']} left")
            col3.markdown('<div class="tooltip">Average Latency<span class="tooltiptext">Average latency across all users (ms).</span></div>', unsafe_allow_html=True)
            col3.metric("", f"{data['avg_latency']:.1f} ms")
            col4.markdown('<div class="tooltip">Network Health<span class="tooltiptext">Overall network health score (0-100).</span></div>', unsafe_allow_html=True)
            col4.metric("", f"{data['health_score']}/100")

            # Highlight Key Insight
            if data['health_score'] >= 90:
                st.markdown(f"<div class='highlight-insight'>🎉 Excellent Network Health Score of {data['health_score']}/100! Most QoS requirements are met.</div>", unsafe_allow_html=True)

            # Efficiency Metrics
            st.subheader("Efficiency Metrics")
            col1, col2, col3 = st.columns(3)
            col1.markdown('<div class="tooltip">Bandwidth Utilization<span class="tooltiptext">Percentage of available bandwidth used.</span></div>', unsafe_allow_html=True)
            col1.metric("", f"{data['metrics']['bw_utilization']:.1f}%")
            col2.markdown('<div class="tooltip">Power Utilization<span class="tooltiptext">Percentage of available power used.</span></div>', unsafe_allow_html=True)
            col2.metric("", f"{data['metrics']['pw_utilization']:.1f}%")
            col3.markdown('<div class="tooltip">Energy Efficiency<span class="tooltiptext">Score reflecting power efficiency (0-100).</span></div>', unsafe_allow_html=True)
            col3.metric("", f"{data['energy_score']}/100")
            st.markdown(f"<div class='info-box'>The bandwidth utilization of {data['metrics']['bw_utilization']:.1f}% and power utilization of {data['metrics']['pw_utilization']:.1f}% indicate how efficiently resources were allocated. An energy efficiency score of {data['energy_score']} reflects the balance between performance and power consumption.</div>", unsafe_allow_html=True)

            # Highlight Key Insight
            if data['metrics']['bw_utilization'] > 90 or data['metrics']['pw_utilization'] > 90:
                st.markdown(f"<div class='highlight-insight'>⚠️ High Utilization Detected: Bandwidth at {data['metrics']['bw_utilization']:.1f}%, Power at {data['metrics']['pw_utilization']:.1f}%. Consider increasing resources.</div>", unsafe_allow_html=True)

            # Detailed Per-User Metrics
            st.subheader("Per-User Metrics")
            detailed_df = data['df'].copy()
            detailed_df["QoS Met"] = detailed_df.apply(lambda row: "Yes" if row["Latency (ms)"] <= user_profiles[int(row["User ID"].split("_")[1]) - 1]["latency"] else "No", axis=1)
            st.dataframe(detailed_df.style.apply(lambda x: ['background: #d4edda' if v == "Yes" else 'background: #f8d7da' for v in x], subset=['QoS Met']).background_gradient(cmap="viridis"))

            # Resource Gauges
            st.subheader("Resource Gauges")
            col5, col6 = st.columns(2)
            fig_gauge_bw = go.Figure(go.Indicator(mode="gauge+number", value=data['total_bw'], domain={'x': [0, 1], 'y': [0, 1]},
                                                  title={'text': "Bandwidth (MHz)"}, gauge={'axis': {'range': [0, data['available_bandwidth']]}}))
            fig_gauge_pw = go.Figure(go.Indicator(mode="gauge+number", value=data['total_pw'], domain={'x': [0, 1], 'y': [0, 1]},
                                                  title={'text': "Power (Watts)"}, gauge={'axis': {'range': [0, data['available_power']]}}))
            col5.plotly_chart(fig_gauge_bw, use_container_width=True, key=f"gauge_bw_{iteration_key}")
            col6.plotly_chart(fig_gauge_pw, use_container_width=True, key=f"gauge_pw_{iteration_key}")

            # Visual Insights (Bar Chart and 3D Scatter Plot only)
            st.subheader("Visual Insights")
            fig = px.bar(data['df'], x="User ID", y=["Bandwidth (MHz)", "Power (Watts)", "Latency (ms)"], barmode="group", title="Resource Allocation per User")
            st.plotly_chart(fig, use_container_width=True, key=f"chart_main_{iteration_key}")

            fig_3d = go.Figure(data=[go.Scatter3d(x=data['df']["Bandwidth (MHz)"], y=data['df']["Power (Watts)"],
                                                  z=data['df']["Latency (ms)"], mode="markers",
                                                  marker=dict(size=8, color=data['df'].index, colorscale="Blues"))])
            fig_3d.update_layout(title="6G Resource Space", scene=dict(xaxis_title="Bandwidth", yaxis_title="Power", zaxis_title="Latency"))
            st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_chart_main_{iteration_key}")

            # 5G vs 6G Comparison
            st.subheader("5G vs 6G Comparison")
            comparison_data = pd.DataFrame({
                "Technology": ["5G", "6G"],
                "Latency (ms)": [10, data['avg_latency']],
                "Bandwidth Efficiency (%)": [70, data['metrics']['bw_utilization']],
                "Energy Efficiency Score": [60, data['energy_score']]
            })
            fig_comparison = px.bar(comparison_data, x="Technology", y=["Latency (ms)", "Bandwidth Efficiency (%)", "Energy Efficiency Score"], barmode="group")
            st.plotly_chart(fig_comparison, use_container_width=True, key=f"comparison_chart_{iteration_key}")
            st.markdown("<div class='info-box'>6G offers lower latency and higher bandwidth efficiency compared to 5G, as shown above. This app leverages 6G's capabilities to achieve better resource allocation and energy efficiency.</div>", unsafe_allow_html=True)

            # Predictive Insights
            st.subheader("Predictive Insights")
            if data['metrics']['bw_utilization'] > 90:
                st.warning("High bandwidth utilization detected. Consider increasing available bandwidth or reducing the number of users in the next cycle.")
            if data['metrics']['pw_utilization'] > 90:
                st.warning("High power utilization detected. Consider increasing available power or optimizing power allocation in the next cycle.")
            if data['avg_latency'] > 20:
                st.warning("Average latency is higher than optimal. Consider prioritizing users with lower latency requirements in the next cycle.")
            else:
                st.success("Resource allocation is optimal for the current cycle.")

            # Alerts
            if data['total_bw'] > data['available_bandwidth'] or data['total_pw'] > data['available_power']:
                st.error("Over-allocation detected!")
            if any(data['df']["Latency (ms)"] > [p["latency"] for p in user_profiles[:len(data['df'])]]):
                st.warning("Latency exceeds QoS requirements for some users!")

            # Download PDF Report
            pdf_buffer = generate_pdf_report(data['df'], data['metrics'], data['energy_score'], data['health_score'], data['raw_response'])
            st.download_button("Download PDF Report", pdf_buffer, f"allocation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf", key=f"download_pdf_{iteration_key}")

            if not continuous_mode:
                st.session_state.running = False
            if stop_clicked:
                st.session_state.running = False
                break
            if st.session_state.running and continuous_mode:
                time.sleep(refresh_rate)

# Understanding the Dashboard
if st.session_state.display_help:
    st.subheader("Know Your Dashboard: A Comprehensive Guide")
    bw_utilization_display = st.session_state.allocation_data['metrics']['bw_utilization'] if st.session_state.allocation_data else default_bw_utilization
    energy_score_display = st.session_state.allocation_data['energy_score'] if st.session_state.allocation_data else default_energy_score
    allocation_df_display = st.session_state.allocation_data['df'] if st.session_state.allocation_data else default_allocation_df
    avg_latency_display = st.session_state.allocation_data['avg_latency'] if st.session_state.allocation_data else 0

    user_15_latency = allocation_df_display['Latency (ms)'].get(14, 'N/A') if 'Latency (ms)' in allocation_df_display.columns and len(allocation_df_display) > 14 else 'N/A'

    st.markdown("""
    ### Introduction to the Dashboard
    The **LLM-Based B5G/6G Resource Allocator** is a cutting-edge tool designed to optimize resource allocation in Beyond 5G (B5G) and 6G wireless networks. Leveraging AI-driven insights, this dashboard helps network engineers and researchers allocate critical resources—bandwidth, power, and latency—to multiple users based on their Quality of Service (QoS) requirements. The app uses the Gemini API to perform intelligent resource allocation, ensuring fairness, efficiency, and performance in next-generation networks.

    #### Purpose and Goals
    - **Optimize Resource Allocation**: Efficiently distribute bandwidth, power, and latency to meet diverse QoS needs in 6G networks.
    - **Leverage AI**: Use large language models (LLMs) to make informed allocation decisions, balancing performance and resource constraints.
    - **Provide Insights**: Offer actionable insights through metrics, visualizations, and predictive recommendations to improve network performance.
    - **Compare Technologies**: Highlight the advantages of 6G over 5G in terms of latency, bandwidth efficiency, and energy efficiency.

    ### Dashboard Components
    The dashboard is organized into several sections, each providing specific insights into the resource allocation process. Below is a detailed breakdown of each component:

    #### 1. Quick Stats
    - **Purpose**: Provides a high-level overview of the network configuration.
    - **Details**:
      - **Total Users**: Displays the number of users configured for allocation (e.g., {num_users}).
      - **Available Bandwidth**: Shows the total bandwidth available for allocation in MHz (e.g., {available_bandwidth} MHz).
      - **Available Power**: Indicates the total power available in Watts (e.g., {available_power} W).
    - **How to Use**: Use this section to confirm your network setup before starting the allocation process.

    #### 2. Allocation Results
    - **Purpose**: Summarizes the overall results of the resource allocation.
    - **Details**:
      - **Total Bandwidth**: Shows the allocated bandwidth out of the available bandwidth (e.g., {total_bw}/{available_bandwidth} MHz).
      - **Total Power**: Displays the allocated power out of the available power (e.g., {total_pw}/{available_power} W).
      - **Average Latency**: Indicates the average latency across all users (e.g., {avg_latency:.1f} ms).
      - **Network Health**: Provides a score (0-100) reflecting the overall health of the network based on QoS satisfaction (e.g., {health_score}/100).
    - **How to Use**: Use this section to quickly assess the overall performance of the allocation. A high network health score indicates that most QoS requirements were met.

    #### 3. Efficiency Metrics
    - **Purpose**: Evaluates the efficiency of the resource allocation.
    - **Details**:
      - **Bandwidth Utilization**: Percentage of available bandwidth used (e.g., {bw_utilization:.1f}%).
      - **Power Utilization**: Percentage of available power used (e.g., {pw_utilization:.1f}%).
      - **Energy Efficiency**: A score (0-100) reflecting the balance between performance and power consumption (e.g., {energy_score}/100).
    - **How to Use**: Monitor these metrics to ensure resources are being used efficiently. High utilization percentages indicate effective use of resources, while a high energy efficiency score suggests sustainable power usage.

    #### 4. Per-User Metrics
    - **Purpose**: Provides a detailed breakdown of resource allocation for each user.
    - **Details**:
      - **User ID**: The identifier for each user (e.g., User_1, User_2).
      - **Bandwidth (MHz)**: Allocated bandwidth for the user.
      - **Power (Watts)**: Allocated power for the user.
      - **Latency (ms)**: Achieved latency for the user.
      - **QoS Requirement**: The user’s QoS level (High, Medium, Low).
      - **QoS Met**: Indicates whether the user’s latency requirement was met (Yes/No, highlighted in green/red).
    - **How to Use**: Review this table to ensure that each user’s QoS requirements are met. If a user’s QoS is not met (red), consider adjusting their settings or increasing available resources.

    #### 5. Resource Gauges
    - **Purpose**: Visualizes the total bandwidth and power usage.
    - **Details**:
      - **Bandwidth Gauge**: Shows the allocated bandwidth relative to the available bandwidth.
      - **Power Gauge**: Displays the allocated power relative to the available power.
    - **How to Use**: Use these gauges to quickly assess resource consumption. If the gauges are near the maximum, you may need to increase available resources.

    #### 6. Visual Insights
    - **Purpose**: Provides interactive visualizations of the allocation.
    - **Details**:
      - **Bar Chart**: Displays bandwidth, power, and latency for each user side by side.
      - **3D Scatter Plot**: Visualizes the allocation in a 3D space with bandwidth, power, and latency as axes.
    - **How to Use**: Use these charts to identify patterns in the allocation. For example, the bar chart can show how resources are distributed across users.

    #### 7. 5G vs 6G Comparison
    - **Purpose**: Compares the performance of 5G and 6G technologies.
    - **Details**:
      - **Latency**: Compares the average latency (e.g., 5G: 10 ms, 6G: {avg_latency:.1f} ms).
      - **Bandwidth Efficiency**: Compares bandwidth utilization (e.g., 5G: 70%, 6G: {bw_utilization:.1f}%).
      - **Energy Efficiency**: Compares energy efficiency scores (e.g., 5G: 60, 6G: {energy_score}).
    - **How to Use**: Use this comparison to understand the advantages of 6G over 5G, such as lower latency and higher efficiency.

    #### 8. Predictive Insights
    - **Purpose**: Offers recommendations for optimizing future allocations.
    - **Details**:
      - Warns if bandwidth or power utilization exceeds 90%.
      - Alerts if average latency is higher than optimal (>20 ms).
      - Confirms if the allocation is optimal.
    - **How to Use**: Follow these recommendations to improve the next allocation cycle. For example, if bandwidth utilization is high, consider adding more bandwidth.

    #### 9. AI Summary
    - **Purpose**: Provides a detailed explanation of the allocation process from the AI.
    - **Details**:
      - Describes how resources were allocated to each user.
      - Explains how QoS requirements influenced the allocation.
      - Highlights any trade-offs made.
      - Discusses the energy efficiency and network health scores.
    - **How to Use**: Read this summary to understand the AI’s decision-making process and identify areas for improvement.

    ### How to Use the Dashboard Effectively
    1. **Configure the Network**:
       - In the sidebar, set the number of users, available bandwidth, and available power.
       - For each user, specify their QoS level (High, Medium, Low), maximum latency, and reliability percentage.
    2. **Start Allocation**:
       - Click "Start Allocation" to perform a one-time allocation.
       - Check "Continuous Mode" to run allocations in real-time (refreshes every few seconds based on the refresh rate).
    3. **Review Results**:
       - Use the "Allocation Results" and "Efficiency Metrics" to assess overall performance.
       - Check "Per-User Metrics" to ensure QoS requirements are met for each user.
       - Use visualizations in "Visual Insights" to identify patterns.
    4. **Optimize Based on Insights**:
       - Follow recommendations in "Predictive Insights" to adjust settings for the next cycle.
       - Review the "AI Summary" for detailed explanations of the allocation.
    5. **View History**:
       - Use the "Toggle History View" button in the sidebar to review past allocations.
       - Select a past allocation to analyze its results and metrics.

    ### Technical Details
    #### AI-Driven Allocation
    - The app uses the Gemini API (model: `gemini-1.5-flash`) to perform resource allocation.
    - The AI takes the following inputs:
      - User IDs and their QoS requirements (latency, reliability, QoS level).
      - Available bandwidth and power.
      - An option to enable AI optimization for more precise allocations.
    - The AI outputs:
      - Resource allocations for each user (bandwidth, power, latency).
      - Energy efficiency and network health scores.
      - A detailed explanation of the allocation process.

    #### 5G vs 6G: Technical Comparison
    - **Latency**: 5G typically achieves 10-20 ms, while 6G targets sub-5 ms. For example, User_15 has a latency of {user_15_latency} ms.
    - **Bandwidth Efficiency**: 6G supports higher efficiency (e.g., {bw_utilization:.1f}% in this allocation) compared to 5G (~70%), allowing more users to be served.
    - **Energy Efficiency**: 6G focuses on sustainability, achieving higher scores (e.g., {energy_score} vs 5G’s typical 60).
    - **Reliability**: 6G offers ultra-reliable low-latency communication (URLLC), supporting reliability up to 99.9999%, compared to 5G’s 99.9%.

    #### Database Storage
    - All allocations are stored in a SQLite database (`allocation_history.db`).
    - Each allocation record includes:
      - Timestamp
      - Allocation data (as JSON)
      - Energy and health scores
      - Available bandwidth and power
      - AI summary
      - Metrics (utilization, latency, etc.)

    ### Tips for Optimizing Resource Allocation
    1. **Balance QoS Requirements**:
       - Assign appropriate QoS levels (High, Medium, Low) based on user needs. High QoS users require more resources, so balance them with Medium/Low QoS users.
    2. **Adjust Available Resources**:
       - If utilization is high (>90%), increase available bandwidth or power to avoid over-allocation.
    3. **Prioritize Latency**:
       - For latency-sensitive applications (e.g., autonomous vehicles), set lower maximum latency values and ensure the AI prioritizes these users.
    4. **Use Continuous Mode for Real-Time Monitoring**:
       - Enable "Continuous Mode" to monitor allocation in real-time, especially in dynamic network environments.
    5. **Analyze Past Allocations**:
       - Use the history view to identify trends and improve future allocations.

    ### Why This Approach?
    - **AI-Driven**: Using an LLM ensures intelligent, adaptive allocation that considers complex QoS requirements and resource constraints.
    - **Real-Time Insights**: The dashboard provides immediate feedback, allowing for quick adjustments.
    - **Scalability**: Designed for 6G networks, the app can handle a large number of users and diverse requirements.
    - **Educational Value**: The detailed AI summary and 5G vs 6G comparison provide learning opportunities for understanding next-generation networks.

    ### Conclusion
    The LLM-Based B5G/6G Resource Allocator is a powerful tool for network optimization, combining AI-driven allocation with comprehensive analytics. By understanding and utilizing each section of the dashboard, you can achieve efficient, fair, and sustainable resource allocation in 6G networks, paving the way for advanced wireless communication systems.
    """.format(
        num_users=num_users,
        available_bandwidth=available_bandwidth,
        available_power=available_power,
        total_bw=st.session_state.allocation_data['total_bw'] if st.session_state.allocation_data else 0,
        total_pw=st.session_state.allocation_data['total_pw'] if st.session_state.allocation_data else 0,
        avg_latency=avg_latency_display,
        health_score=st.session_state.allocation_data['health_score'] if st.session_state.allocation_data else 0,
        bw_utilization=bw_utilization_display,
        pw_utilization=st.session_state.allocation_data['metrics']['pw_utilization'] if st.session_state.allocation_data else 0,
        energy_score=energy_score_display,
        user_15_latency=user_15_latency
    ), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666666; font-size: 14px; margin-top: 15px;'>Developed by KK | Powered by 6G Innovation</p>", unsafe_allow_html=True)