import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ==========================================
# 1. DASHBOARD SETUP
# ==========================================

st.set_page_config(
    page_title="AI for Predictive Maintenance Demo",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""

## 🏢 AI-based Product for Predictive Maintenance Applications

**Prototype Goal:** Demonstrate real-time anomaly detection using **Unsupervised Machine Learning**.
This model learns a precise non-linear boundary around 'healthy' equipment states.
When the equipment operates outside this boundary, an alert is triggered with specific maintenance recommendations.

*Built by [Raka Adrianto](https://www.linkedin.com/in/lugasraka/?) | December 2025*

---

## 📖 How to Use This Demo

This interactive application simulates a **real-time equipment monitoring system** for your facility. Use the sliders on the left to adjust sensor readings and observe how the machine's health margin changes. 

**Key Features:**
- **Live Sensor Simulation:** Adjust temperature, speed, torque, and wear in real time
- **Anomaly Detection:** The SVM model identifies abnormal operating patterns instantly
- **Health Gauge:** Visualizes how close the equipment is to failure
- **Actionable Alerts:** Receive specific maintenance recommendations when anomalies are detected
- **Feature Importance:** Understand which sensor readings triggered the alert

---
""")

# Dataset information collapsed by default to keep dashboard focused
with st.expander("📚 Dataset & Experimentations", expanded=False):
    st.markdown("""
    This demo uses the AI4I 2020 Predictive Maintenance dataset from the UCI Machine Learning Repository. The dataset contains simulated sensor readings from manufacturing equipment and includes:

    - Air temperature (K)
    - Process temperature (K)
    - Rotational speed (RPM)
    - Torque (Nm)
    - Tool wear (min)
    - A labeled failure column (used only for analysis; the SVM below is trained unsupervised)

    Source & details:
    [Explainable Artificial Intelligence for Predictive Maintenance Applications](https://ieeexplore.ieee.org/document/9253083) by S. Matzka. 2020
                
    **Experimentation Notes:**
    - **[Isolation Forest](https://en.wikipedia.org/wiki/Isolation_forest)**: An ensemble method that isolates anomalies by randomly selecting features and split values, effective for high-dimensional data.
    - **[Local Outlier Factor](https://en.wikipedia.org/wiki/Local_outlier_factor)**: A density-based algorithm that compares the local density of points to identify local outliers in a dataset.
    - **[One-Class SVM](https://en.wikipedia.org/wiki/Support_vector_machine#One-class_SVM)**: Learns a boundary around normal data points and flags anything outside as anomalous, excellent at capturing non-linear patterns.
    
    One-Class SVM provided the best balance of detection accuracy and interpretability for this dataset. For data enthusiasts, feel free to modify the model parameters in the [Jupyter notebook](https://colab.research.google.com/drive/1IMA_gA589DMr5fxneNDWjxmidPm0r2kK?usp=sharing)
    """)

# Explanation for the User
with st.expander("ℹ️ How to Simulate Failures - Scenario Guide"):
    st.markdown("""
        ### Quick Start Guide
        
        **How the Model Works:**
        - We are running a **One-Class SVM** that learns the boundary of healthy equipment states
        - Any reading **outside** this boundary triggers an anomaly alert
        - SVM is excellent at detecting **subtle combinations** of bad conditions (e.g., high torque + low RPM)
        
        ### Try These Failure Scenarios:
        
        #### **Scenario 1: High Torque Failure** ⚙️
        1. Move **Torque** slider to **70 Nm** (keep other readings normal)
        2. Watch the gauge turn red and anomaly detection trigger
        3. → The system detects excessive mechanical load
        
        #### **Scenario 2: Critical Combination - Stress Lock** 💪⚡
        1. Set **Torque** to **55 Nm**
        2. Set **RPM** to **1200** (low speed)
        3. → This high-load, low-speed combination is a classic failure pattern
        4. Recommendation: Increase speed to relieve pressure OR reduce load
        
        #### **Scenario 3: High Wear + Temperature** ⏱️🔥
        1. Move **Tool Wear** slider to **200+ min**
        2. Set **Process Temperature** to **313 K**
        3. → Combined thermal stress and mechanical wear accelerate failure
        4. Recommendation: Replace tool immediately AND cool the system
        
        #### **Scenario 4: Motor Struggling** ⚡💪
        1. Set **RPM** to **1150** (low)
        2. Set **Torque** to **55 Nm** (high)
        3. → Motor can't handle the load at low speed
        4. Recommendation: Check bearings, alignment, and lubrication
        
        #### **Scenario 5: Thermal Overload** 🔥
        1. Set **Process Temperature** to **314 K** (high)
        2. Increase **Air Temperature** to **304 K**
        3. Increase **Torque** to **50 Nm**
        4. → Multiple stress factors compound to create failure conditions
        
        ### Understanding the Output:
        - **Green Gauge (Positive Score):** Equipment is healthy ✅
        - **Red Gauge (Negative Score):** Equipment has crossed failure threshold ⚠️
        - **Warnings Section:** Specific issues detected and their severity
        - **Recommendations Section:** Actionable steps your maintenance team should take
        
        ### Tips for Facility Managers:
        - Use this app to **train staff** on equipment warning signs
        - The recommended actions are **specific to your equipment** – implement them immediately
        - **Proactive maintenance** (before alerts) is more cost-effective than reactive repairs
        - Monitor **tool wear** closely – it's often the easiest maintenance task to schedule
        """)


# ==========================================
# 2. MODEL TRAINING (One-Class SVM)
# ==========================================
@st.cache_resource
def train_model():
    # Load the AI4I dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    try:
        df = pd.read_csv(url)
    except:
        st.error("Could not load data. Check internet connection or download the file locally.")
        return None, None

    # Rename and Prep
    df.rename(columns={
        'Air temperature [K]': 'Air_Temp',
        'Process temperature [K]': 'Process_Temp',
        'Rotational speed [rpm]': 'RPM',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Wear_Time'
    }, inplace=True)

    features = ['Air_Temp', 'Process_Temp', 'RPM', 'Torque', 'Wear_Time']
    X = df[features]

    # Scale Data
    # CRITICAL FOR SVM: SVM calculates distances, so scaling is mandatory.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train One-Class SVM
    # nu=0.03 approximates the contamination (outlier fraction)
    # kernel='rbf' allows it to learn non-linear relationships (curved boundaries)
    print("Training SVM Model... (This might take a few seconds)")
    model = OneClassSVM(nu=0.03, kernel="rbf", gamma='scale') 
    model.fit(X_scaled)

    return model, scaler

with st.spinner('Initializing One-Class SVM Model...'):
    model, scaler = train_model()

# ==========================================
# 3. SIDEBAR: LIVE SENSOR SIMULATION
# ==========================================
st.sidebar.header("🎛️ Live Sensor Feed")
st.sidebar.markdown("""
Adjust the sliders below to simulate real equipment readings. 
Monitor how each parameter affects the system health.
""")

# Define ranges based on the real dataset
air_temp = st.sidebar.slider("🌡️ Air Temperature [K]", 295.0, 305.0, 300.0, 
                              help="Ambient room temperature. Higher temps increase stress.")
proc_temp = st.sidebar.slider("🔥 Process Temperature [K]", 305.0, 315.0, 310.0,
                               help="Equipment operating temperature. Overheating is a failure indicator.")
rpm = st.sidebar.slider("⚡ Rotational Speed [RPM]", 1100, 2900, 1500,
                        help="Motor speed. Very low RPM with high torque = high stress combination.")
torque = st.sidebar.slider("💪 Torque [Nm]", 0.0, 80.0, 40.0,
                           help="Force/load on equipment. High torque accelerates wear.")
wear = st.sidebar.slider("⏱️ Tool Wear [min]", 0, 260, 0,
                         help="Cumulative wear time. Higher wear = higher failure risk.")

# Create a dataframe for the "Live" single point
input_data = pd.DataFrame({
    'Air_Temp': [air_temp],
    'Process_Temp': [proc_temp],
    'RPM': [rpm],
    'Torque': [torque],
    'Wear_Time': [wear]
})

# ==========================================
# 4. INFERENCE & LOGIC
# ==========================================
def generate_recommendations(air_temp, proc_temp, rpm, torque, wear, is_anomaly, score):
    """Generate actionable maintenance recommendations based on sensor readings."""
    recommendations = []
    warnings = []
    
    # Temperature checks
    if proc_temp > 312:
        warnings.append("🔥 **High Process Temperature** - Equipment running hotter than normal. Risk of thermal degradation.")
        recommendations.append("✓ Check cooling system and ventilation. Inspect for dust/blockages in cooling fans.")
    
    if air_temp > 302:
        warnings.append("🌡️ **High Ambient Temperature** - Room temperature elevated. Facility cooling may be inadequate.")
        recommendations.append("✓ Improve facility air conditioning. Reduce ambient temperature to <302 K.")
    
    # RPM checks
    if rpm < 1300:
        warnings.append("⚡ **Low Rotational Speed** - Motor running slower than optimal. Check for bearing friction.")
        recommendations.append("✓ Inspect motor bearings and lubrication. Verify power supply voltage is stable.")
    
    if rpm > 2700:
        warnings.append("⚡ **High Rotational Speed** - Motor exceeding typical operating range. Increased vibration risk.")
        recommendations.append("✓ Monitor vibration levels. Check for loose mechanical components.")
    
    # Torque checks
    if torque > 60:
        warnings.append("💪 **Excessive Torque** - Load far exceeds normal operating range. High stress on gearbox/shaft.")
        recommendations.append("✓ **URGENT:** Reduce load immediately. Check for mechanical jamming or binding.")
    
    if torque > 50:
        warnings.append("💪 **High Torque Load** - Equipment under elevated mechanical stress.")
        recommendations.append("✓ Monitor gearbox and shaft alignment. Consider load balancing.")
    
    # Wear checks
    if wear > 200:
        warnings.append("⏱️ **Critical Tool Wear** - Tool life nearly exhausted. Imminent failure risk.")
        recommendations.append("✓ **SCHEDULE REPLACEMENT:** Replace tool immediately to prevent catastrophic failure.")
    
    if wear > 150:
        warnings.append("⏱️ **High Tool Wear** - Significant degradation detected. Performance may degrade.")
        recommendations.append("✓ Schedule tool replacement within next maintenance window (within 24-48 hours).")
    
    if wear > 100:
        warnings.append("⏱️ **Moderate Tool Wear** - Normal wear accumulation. Monitor closely.")
        recommendations.append("✓ Plan tool replacement in next scheduled maintenance cycle.")
    
    # Combination anomalies
    if torque > 50 and rpm < 1500:
        warnings.append("⚠️ **CRITICAL COMBINATION:** High load + low speed = extreme stress. This is a classic failure pattern.")
        recommendations.append("✓ **IMMEDIATE ACTION:** Reduce torque below 40 Nm OR increase RPM above 1800 to relieve stress.")
    
    if proc_temp > 310 and wear > 150:
        warnings.append("⚠️ **Combined Thermal + Wear Stress:** High temperature + high wear accelerates failure.")
        recommendations.append("✓ **Dual Action:** Cool equipment AND schedule tool replacement urgently.")
    
    if rpm < 1300 and torque > 50:
        warnings.append("⚠️ **Struggling Motor:** Slow speed with high load indicates mechanical resistance.")
        recommendations.append("✓ Inspect for bearing wear, shaft misalignment, or internal resistance. Lubricate moving parts.")
    
    # Overall anomaly message
    if is_anomaly:
        warnings.insert(0, "🚨 **ANOMALY DETECTED** - Equipment operating outside normal parameters!")
        recommendations.insert(0, "⚠️ **Alert your maintenance team immediately.** Do not ignore this warning.")
    
    return warnings, recommendations

if model:
    # Scale the input using the same scaler as training
    input_scaled = scaler.transform(input_data)

    # Get Anomaly Score
    # SVM Decision Function: Positive = Normal (Inside boundary), Negative = Anomaly (Outside)
    score = model.decision_function(input_scaled)[0]
    prediction = model.predict(input_scaled)[0] # 1 is normal, -1 is anomaly

    is_anomaly = prediction == -1

    # Get recommendations
    warnings, recommendations = generate_recommendations(air_temp, proc_temp, rpm, torque, wear, is_anomaly, score)

    # ==========================================
    # 5. MAIN DASHBOARD VISUALS
    # ==========================================
    
    # Top Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("System Status")
        if is_anomaly:
            st.error("⚠️ CRITICAL ANOMALY DETECTED")
        else:
            st.success("✅ SYSTEM NORMAL")
            
    with col2:
        # SVM scores are distances to the margin. 
        # Large positive = Very Safe. Negative = Anomaly.
        st.metric(label="Margin Distance (SVM Score)", value=f"{score:.2f}")
    
    with col3:
        st.metric(label="Calculated Stress Load", value=f"{(torque * rpm / 5252):.1f} HP")

    st.divider()

    # Gauge Chart for Visual Impact
    # Adjusted range for SVM scores which typically range from -1 to +1 or higher
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Health Margin (Negative = Failure)"},
        gauge = {
            'axis': {'range': [-1, 1]}, # SVM scores have a wider range than IsoForest
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, 0], 'color': "salmon"}, # Danger Zone
                {'range': [0, 1], 'color': "lightgreen"}], # Safe Zone
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Display warnings and recommendations
    if warnings or recommendations:
        # Prominent red alert when the SVM flags an anomaly
        if is_anomaly:
            st.error("🚨 ANOMALY DETECTED - Equipment operating outside normal parameters!", icon="🚨")
        else:
            st.subheader("⚠️ Diagnostic Analysis")
        
        col_warn, col_rec = st.columns(2)
        
        with col_warn:
            st.markdown("### Detected Issues:")
            for warning in warnings:
                if 'ANOMALY DETECTED' in warning or 'CRITICAL' in warning:
                    st.error(warning)
                else:
                    st.warning(warning, icon="⚠️")
        
        with col_rec:
            st.markdown("### Recommended Actions:")
            for rec in recommendations:
                # Show urgent recommendations as error-style for visibility
                if rec.startswith('⚠️') or rec.startswith('✓ **URGENT'):
                    st.error(rec)
                else:
                    st.info(rec, icon="✅")
    else:
        st.success("✅ No issues detected. Equipment operating within normal parameters.")


    st.divider()
