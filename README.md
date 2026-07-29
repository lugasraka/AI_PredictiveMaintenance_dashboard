# 🏢 AI for Predictive Maintenance

A Streamlit dashboard that flags failing industrial equipment using a One-Class SVM trained on the AI4I 2020 dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-predictive-maintenance-dashboard.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 [Try the Live Demo](https://ai-predictive-maintenance-dashboard.streamlit.app/)

## Contents

- [What it is](#what-it-is)
- [Who it's for](#who-its-for)
- [Why it matters](#why-it-matters)
- [How it works](#how-it-works)
- [Run it locally](#run-it-locally)

## What it is

The dashboard watches five sensor readings — air temperature, process temperature, rotational speed, torque, and tool wear — and flags records that look unlike normal operation. A health gauge shows how close the current reading is to the failure boundary, and a feature panel points at the sensor most likely to blame.

It's built with Streamlit, Plotly, and scikit-learn, and runs on the AI4I 2020 dataset from the UCI Machine Learning Repository. The trained model is bundled with the app, so the demo loads with no setup.

## Who it's for

![User Personas](2%20Personas.png)

**Facility managers** running plants or large buildings with critical HVAC and motor equipment. They want to know which assets are likely to fail in the next few weeks, not which ones passed inspection last month. The dashboard gives them a live read on equipment health and a short list of what to look at first.

**Maintenance technicians** who get sent to fix things. They want a clear, specific signal before they walk up to the machine — "high torque, low RPM, check bearings" is more useful than "anomaly detected." The dashboard's feature attribution panel is built around that use case.

## Why it matters

Unplanned downtime is expensive. A single motor failure on a production line can cost tens of thousands of dollars in lost output, emergency repair labor, and cascading damage to other equipment. Preventive maintenance helps, but it sends people to inspect machines that don't need attention and still misses failures that happen between scheduled visits.

Predictive maintenance sits in the middle. Sensors are already on the equipment. The question is whether anyone is reading them with the right model.

A rough estimate for a mid-sized facility with 500 critical assets:

| Metric | Before | After |
|---|---|---|
| Unplanned downtime per year | 8–12 incidents | 2–4 incidents |
| Emergency repair costs | ~$150K | ~$40K |
| Net annual benefit | — | ~$200K |

These numbers are illustrative — actual savings depend on the facility, the equipment, and how well the alerts feed into the maintenance workflow.

## How it works

**Dataset.** The AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository (Matzka, 2020). 10,000 records, five physical sensor features, and a ground-truth failure label used only for evaluation. Failures are rare — about 3.4% of the data — which is why a supervised classifier trained on the raw labels would overfit to "no failure" almost every time.

**Approach.** Unsupervised anomaly detection. The model sees only healthy records during training and learns the boundary of normal operation. Anything outside that boundary is flagged. This avoids the label problem entirely and works on equipment with no failure history.

Three algorithms were tested: Isolation Forest, Local Outlier Factor, and One-Class SVM with an RBF kernel. All three were trained with a contamination parameter of 3.4% to match the expected outlier rate. Predictions were scored against the held-out ground truth labels.

**Why One-Class SVM.** It caught the most failures without drowning the user in false alarms.

| Model | Recall | Precision | F1 | Train time |
|---|---|---|---|---|
| Isolation Forest | 0.635 | 0.115 | 0.195 | ~0.3s |
| Local Outlier Factor | 0.561 | 0.132 | 0.214 | ~0.5s |
| **One-Class SVM** | **0.694** | **0.139** | **0.232** | ~1.2s |

The RBF kernel picks up non-linear failure patterns (high torque combined with low RPM, for example) that the tree-based methods miss. The recall is high enough to be useful in a real maintenance workflow, and the training cost is a one-time hit on a 10,000-row dataset.

## Run it locally

**1. Clone the repo**

```bash
git clone http://github.com/lugasraka/AI_PredictiveMaintenance_dashboard.git
cd AI_PredictiveMaintenance_dashboard
```

**2. Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\Activate.ps1  # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the dashboard**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**5. (Optional) Open the notebook**

```bash
jupyter notebook AI4PredictiveBuildings.ipynb
```

The notebook has the full model exploration — feature scaling, parameter tuning, and the comparison plots behind the table above.

### Project structure

```
AI_PredictiveMaintenance_dashboard/
├── app.py                          # Streamlit dashboard
├── AI4PredictiveBuildings.ipynb    # Model experimentation
├── ai4i2020.csv                    # Dataset
├── requirements.txt                # Python dependencies
├── README.md
└── 2 Personas.png                  # User persona illustrations
```

## Notes

The model is trained on one public dataset and reflects the sensor ranges in that data. It won't transfer to a different machine class without retraining. Sensor drift, concept drift over months of operation, and novel failure modes the training data doesn't cover are real limits — a production deployment would need periodic retraining and a feedback loop from the maintenance team.

## Contact

Built by [Raka Adrianto](https://www.linkedin.com/in/lugasraka/?) — December 2025.

Licensed under MIT. Contributions and issues welcome.
