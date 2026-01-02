# 🏢 AI for Predictive Maintenance

**An intelligent real-time anomaly detection system for industrial equipment using Unsupervised Machine Learning**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-predictive-maintenance-dashboard.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 [**Try the Live Demo**](https://ai-predictive-maintenance-dashboard.streamlit.app/)

---

## 📖 Table of Contents

- [Product Overview](#-product-overview)
- [Target Users & Personas](#-target-users--personas)
- [Problem Statement](#-problem-statement)
- [Methodology & Development Steps](#-methodology--development-steps)
- [Model Performance & Trade-offs](#-model-performance--trade-offs)
- [Business Insights & Implications](#-business-insights--implications)
- [Getting Started](#-getting-started)
- [How to Clone & Run](#-how-to-clone--run)

---

## 🎯 Product Overview

**AI for Predictive Maintenance** is an intelligent monitoring system that detects equipment anomalies before failures occur, enabling proactive maintenance scheduling and reducing operational downtime. Built on state-of-the-art **One-Class Support Vector Machine (SVM)** algorithms, the system learns the "healthy" operational patterns of industrial equipment and raises alerts when deviations indicate potential failures.

### Key Features

- **Real-Time Anomaly Detection**: Continuous monitoring of 5 critical sensor parameters (temperature, speed, torque, wear)
- **Unsupervised Learning**: No historical failure labels required—the model learns normal patterns autonomously
- **Actionable Insights**: Specific maintenance recommendations based on detected anomaly patterns
- **Interactive Dashboard**: Live simulation environment for testing different failure scenarios
- **Health Margin Visualization**: Intuitive gauge showing proximity to failure threshold
- **Feature Importance Analysis**: Understanding which sensor triggered the anomaly

### Technology Stack

- **ML Framework**: Scikit-learn (One-Class SVM)
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Cloud-ready (Streamlit Cloud compatible)

---

## 👥 Target Users & Personas

This product serves two primary user groups in industrial and building management environments:

![User Personas](2%20Personas.png)

### Persona 1: Facility Manager (Operations Lead)

**Background**: Oversees day-to-day operations of manufacturing plants or large building facilities with critical HVAC/motor equipment.

**Pain Points**:
- Unexpected equipment failures disrupt production schedules
- Difficulty prioritizing maintenance tasks across multiple assets
- High costs from emergency repairs vs. planned maintenance
- Lack of visibility into equipment health status

**Goals**:
- Minimize unplanned downtime
- Optimize maintenance resource allocation
- Extend equipment lifespan through proactive care
- Reduce operational costs

**How This Product Helps**: Provides real-time alerts with specific maintenance recommendations, enabling proactive scheduling before failures occur. The dashboard helps prioritize which equipment needs immediate attention.

### Persona 2: Maintenance Technician (Field Operator)

**Background**: Hands-on technical staff responsible for equipment inspection, repair, and preventive maintenance.

**Pain Points**:
- Reactive "firefighting" rather than planned work
- Unclear diagnostic information about equipment issues
- Inefficient time allocation checking healthy equipment
- Limited guidance on root cause of failures

**Goals**:
- Receive clear, actionable maintenance tasks
- Understand specific equipment issues before arriving on-site
- Focus efforts on equipment that truly needs attention
- Prevent cascading failures from early intervention

**How This Product Helps**: Delivers specific anomaly explanations (e.g., "High torque + low RPM detected") with targeted recommendations (e.g., "Check bearings and lubrication"), enabling faster diagnosis and more effective repairs.

---

## ❗ Problem Statement

### Business Context

Industrial equipment failures cost businesses **billions annually** through unplanned downtime, emergency repairs, and production losses. Traditional maintenance approaches fall into two categories, both with significant drawbacks:

1. **Reactive Maintenance** (Fix when broken):
	- ❌ Leads to unexpected failures and production stoppages
	- ❌ Higher repair costs from cascading damage
	- ❌ Safety risks from sudden equipment malfunction

2. **Preventive Maintenance** (Fixed schedule inspections):
	- ❌ Wastes resources inspecting healthy equipment
	- ❌ May miss failures that occur between scheduled checks
	- ❌ One-size-fits-all approach doesn't account for actual usage patterns

### The Solution: Predictive Maintenance

**Predictive maintenance** uses real-time sensor data and machine learning to detect equipment anomalies **before** failures occur, enabling:

✅ **Proactive intervention** at optimal times  
✅ **Reduced downtime** through planned maintenance windows  
✅ **Cost savings** by preventing catastrophic failures  
✅ **Extended equipment lifespan** through timely care  
✅ **Data-driven decisions** replacing guesswork  

### Technical Challenge

The core technical challenge is **detecting anomalies in unlabeled data**. Most equipment operates normally, with failures being rare events (3-5% of operational time). Traditional supervised learning requires extensive labeled failure examples, which are:
- Time-consuming and expensive to collect
- Often unavailable for new equipment
- Highly imbalanced (99%+ normal vs <1% failure)

**Our approach**: Use **unsupervised anomaly detection** algorithms that learn the "normal" operational boundary from healthy data, then flag any deviations without requiring failure labels.

---

## 🔬 Methodology & Development Steps

### Dataset

**Source**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)  
**Citation**: Matzka, S. (2020). "Explainable Artificial Intelligence for Predictive Maintenance Applications" - IEEE ICPHM  
**Size**: 10,000 sensor records from manufacturing equipment  
**Features**:
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (RPM)
- Torque (Nm)
- Tool Wear (minutes)
- Machine Failure (ground truth for validation only)

### Development Pipeline

#### Step 1: Data Ingestion & Exploration
- Loaded UCI Machine Learning Repository dataset
- Analyzed feature distributions and correlations
- Identified normal operational ranges for each sensor
- **Key Insight**: Failures constitute only 3.4% of records—highly imbalanced

#### Step 2: Feature Engineering
- Selected 5 physically meaningful sensor features
- Applied **StandardScaler** normalization (critical for SVM distance calculations)
- Created scaled feature space for model training

#### Step 3: Model Selection & Experimentation

Evaluated **three unsupervised anomaly detection algorithms**:

| Algorithm | Approach | Strengths |
|-----------|----------|-----------|
| **Isolation Forest** | Ensemble tree-based isolation | Fast training, handles high dimensions |
| **Local Outlier Factor (LOF)** | Density-based comparison | Detects local anomalies effectively |
| **One-Class SVM** | Boundary learning with kernel trick | Captures non-linear patterns, robust decision boundaries |

**Training Configuration**:
- Contamination parameter: 3.4% (estimated outlier fraction)
- SVM kernel: RBF (Radial Basis Function) for non-linear boundaries
- Cross-validation on 10,000 records

#### Step 4: Model Evaluation

Compared models against ground truth labels (used only for validation):
- **Recall**: What % of actual failures did we catch? (Minimize false negatives)
- **Precision**: Of flagged anomalies, how many were real failures? (Minimize false alarms)
- **F1 Score**: Harmonic mean balancing recall and precision
- **Training Time**: Computational efficiency

#### Step 5: Deployment & Interface Design

- Built interactive Streamlit dashboard
- Implemented real-time prediction pipeline
- Created scenario-based testing environment
- Added interpretable visualizations (health gauge, feature importance)

---

## 📊 Model Performance & Trade-offs

### Benchmark Results

| Model | Recall (Catch Rate) | Precision (Reliability) | F1 Score | Training Time |
|-------|---------------------|-------------------------|----------|---------------|
| **Isolation Forest** | 0.635 | 0.115 | 0.195 | ~0.3s |
| **Local Outlier Factor** | 0.561 | 0.132 | 0.214 | ~0.5s |
| **One-Class SVM** ⭐ | **0.694** | **0.139** | **0.232** | ~1.2s |

### Why One-Class SVM Was Selected

**One-Class SVM** achieved the best balance across key metrics:

✅ **Highest Recall (69.4%)**: Catches nearly 70% of real failures—critical for safety and downtime prevention  
✅ **Best F1 Score (0.232)**: Superior balance between catching failures and avoiding false alarms  
✅ **Non-Linear Boundary**: RBF kernel captures complex failure patterns (e.g., high torque + low RPM)  
✅ **Acceptable Training Time**: 1.2 seconds is negligible for batch retraining  

### Performance Trade-offs

#### False Positives vs. False Negatives

In predictive maintenance, there's an inherent trade-off between **false alarms** and **missed failures**:

| Metric | Impact | Business Cost |
|--------|--------|---------------|
| **False Positive** (unnecessary alert) | Wasted inspection effort | Low - ~$200 labor cost |
| **False Negative** (missed failure) | Unplanned downtime + repair | High - ~$50,000+ per incident |

**Our approach**: Tune for **higher recall** (catch more failures) at the expense of some false positives. The cost of a missed failure far exceeds the cost of extra inspections.

#### Model Interpretability

- **SVM Decision Scores**: Distance from learned boundary indicates failure severity
- **Feature Analysis**: Identify which sensor(s) triggered the anomaly
- **Threshold Tuning**: Adjust sensitivity based on business tolerance for false alarms

#### Limitations & Considerations

⚠️ **Cold Start**: Model requires initial training data representing normal operations  
⚠️ **Concept Drift**: Equipment behavior changes over time—periodic retraining needed  
⚠️ **Sensor Reliability**: Assumes sensor readings are accurate (GIGO principle)  
⚠️ **Novel Failures**: May miss failure modes not represented in training boundary  

---

## 💼 Business Insights & Implications

### ROI Analysis

**Typical Manufacturing Facility (500 critical assets)**

| Metric | Before AI | After AI | Improvement |
|--------|-----------|----------|-------------|
| **Unplanned Downtime** | 8-12 incidents/year | 2-4 incidents/year | **70% reduction** |
| **Average Downtime Cost** | $50,000/incident | $50,000/incident | - |
| **Total Downtime Loss** | $500,000/year | $150,000/year | **$350K savings** |
| **Maintenance Labor** | $200,000/year | $250,000/year | +$50K (proactive work) |
| **Emergency Repair Costs** | $150,000/year | $40,000/year | **$110K savings** |
| **Net Annual Benefit** | - | - | **~$410,000** |

**Payback Period**: Typically **3-6 months** for mid-sized facilities

### Strategic Implications

#### 1. **Operational Excellence**
- Shift from reactive "firefighting" to proactive planning
- Optimize spare parts inventory based on predicted needs
- Schedule maintenance during planned downtime windows

#### 2. **Safety & Compliance**
- Reduce risk of catastrophic failures causing injuries
- Demonstrate due diligence for regulatory compliance
- Maintain audit trails of equipment health monitoring

#### 3. **Asset Lifecycle Management**
- Extend equipment lifespan by 20-30% through timely intervention
- Data-driven capital planning for replacements
- Optimize maintenance vs. replace decisions

#### 4. **Scalability & Integration**
- Easily extends to new equipment types with minimal retraining
- Integrates with existing CMMS (Computerized Maintenance Management Systems)
- Cloud-native architecture supports multi-site deployments

### Key Performance Indicators (KPIs)

Track these metrics to measure product success:

- **Mean Time Between Failures (MTBF)**: Target 30-50% increase
- **Maintenance Cost per Asset**: Target 15-25% reduction
- **Unplanned Downtime Hours**: Target 60-80% decrease
- **Alert Accuracy**: Monitor false positive rate (<30% is acceptable)
- **Response Time**: From alert to maintenance action (<24 hours)

### Competitive Advantages

1. **No Historical Failure Data Required**: Unlike supervised models, works from day one
2. **Real-Time Monitoring**: Continuous assessment vs. periodic inspections
3. **Explainable Predictions**: Operators understand *why* an alert was raised
4. **Low Implementation Barrier**: No complex sensor installations—uses existing IoT data

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for dataset download)

### Quick Demo

Try the live dashboard: [Launch App](https://ai-predictive-maintenance-dashboard.streamlit.app/) *(or run locally below)*

---

## 📥 How to Clone & Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Predictive-Maintenance.git
cd AI-Predictive-Maintenance
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages** (see `requirements.txt`):
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib
- seaborn

### Step 4: Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 5: Explore the Jupyter Notebook (Optional)

To understand the model development process:

```bash
jupyter notebook AI4PredictiveBuildings.ipynb
```

### Project Structure

```
AI-Predictive-Maintenance/
├── app.py                          # Streamlit dashboard (main application)
├── AI4PredictiveBuildings.ipynb    # Model experimentation notebook
├── ai4i2020.csv                    # Dataset (auto-downloaded if missing)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── 2 Personas.png                  # User persona illustrations
└── .venv/                          # Virtual environment (not tracked in git)
```

### Troubleshooting

**Issue**: `ModuleNotFoundError` when running app  
**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

**Issue**: Dataset download fails  
**Solution**: Download manually from [UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv) and place in project root

**Issue**: Streamlit port already in use  
**Solution**: Specify different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📚 Additional Resources

- **Academic Paper**: [Matzka, S. (2020). Explainable AI for Predictive Maintenance](https://ieeexplore.ieee.org/document/9253083)
- **Dataset Documentation**: [UCI ML Repository - AI4I 2020](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- **One-Class SVM**: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

---

## 📧 Contact & Contributions

**Built by**: [Raka Adrianto](https://www.linkedin.com/in/lugasraka/?)  
**Date**: December 2025  
**Project Type**: AI Product Management Portfolio Demonstration

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or reach out directly.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the AI4I 2020 dataset
- Siemens AG for domain knowledge and inspiration
- Streamlit for the excellent dashboard framework

---

**⭐ If you found this project helpful, please consider giving it a star!**




