# 🧠 Parkinson's Disease Prediction using Machine Learning

A machine learning web application that predicts whether a person shows signs of **Parkinson's disease** based on biomedical voice measurements, using a **Support Vector Machine (SVM)** classifier trained on the Oxford Parkinson's Disease Detection Dataset.

**🚀 Live App:** [parkinson-disease-prediction.streamlit.app](https://parkinson-disease-prediction.streamlit.app/)

---

## 📖 Overview

Parkinson's disease is a progressive nervous system disorder that affects movement and often causes measurable changes in a person's voice — including variations in frequency, amplitude, and noise-to-tone ratio — long before other symptoms become obvious. This project uses **22 quantitative voice-measurement features** extracted from sustained vowel phonations to classify whether a subject is healthy or shows signs of Parkinson's disease.

The app supports two modes:
- 🧍 **Single Patient Prediction** — manually enter voice measurement values and get an instant prediction.
- 📄 **Batch Prediction** — upload a CSV of multiple patients and download predictions for all of them at once.

> ⚠️ **Disclaimer:** This tool is built for educational and demonstrative purposes only. It is **not** a certified medical diagnostic tool. Always consult a qualified medical professional for an actual diagnosis.

---

## 🗂️ Dataset

The model is trained on the **Oxford Parkinson's Disease Detection Dataset** (`parkinsons.data`), created by Max Little of the University of Oxford, in collaboration with the National Centre for Voice and Speech, Denver, Colorado.

| Property | Value |
|---|---|
| Instances | 195 voice recordings from 31 people (23 with Parkinson's) |
| Attributes | 23 (22 features + 1 target) |
| Target | `status` — `1` = Parkinson's & `0` = Healthy |
| Missing values | None |

**Feature groups used for prediction:**

| Group | Features |
|---|---|
| Fundamental Frequency | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` |
| Jitter (frequency variation) | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` |
| Shimmer (amplitude variation) | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` |
| Noise-to-Tone Ratios | `NHR`, `HNR` |
| Nonlinear Dynamical Measures | `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE` |

---

## 🔬 Methodology

1. **Exploratory Data Analysis (EDA)** — distribution plots, correlation heatmaps, and class-wise comparisons (e.g. NHR, HNR, RPDE by status) to understand feature relationships and skewness.
2. **Preprocessing** — dropped the non-predictive `name` column, split data into features (`X`) and target (`Y`), then into train/test sets (70/30 split, `random_state=101`).
3. **Model Benchmarking** — trained and evaluated multiple classifiers to compare performance:

   | Model | Test Accuracy | Kappa Score |
   |---|---|---|
   | Gaussian Naïve Bayes | 67.8% | 0.41 |
   | K-Nearest Neighbors | 91.5% | 0.77 |
   | Decision Tree | 91.5% | 0.79 |
   | Logistic Regression | 91.5% | – |
   | Random Forest | 93.2% | – |
   | **Support Vector Machine (linear kernel)** | **91.5%** | **0.77** |

4. **Final Model Selection** — a **linear-kernel SVM** was selected for deployment, achieving **91.5% test accuracy** with a strong **97.7% recall** on Parkinson's-positive cases — prioritizing sensitivity, which matters most in a health-screening context (minimizing missed positive cases).
5. **Serialization** — the trained model was serialized with `pickle` as `deploy_SVM.pkl` for deployment.

---

## 🛠️ Tech Stack

- **Python** — core language
- **scikit-learn** — model training (SVM, Logistic Regression, Random Forest, Decision Tree, Naive Bayes, KNN)
- **pandas / NumPy** — data manipulation
- **seaborn / matplotlib** — exploratory data visualization
- **Streamlit** — interactive web app framework
- **pickle** — model serialization

---

## 📁 Project Structure

```
Parkinson-Disease-Prediction-using-ML-DS/
├── app.py                  # Streamlit web application
├── Parkinson.ipynb         # EDA, model training & comparison notebook
├── parkinsons.data         # Oxford Parkinson's Disease dataset (CSV)
├── ParkinsonNames.txt      # Dataset description & attribute information
├── deploy_SVM.pkl          # Final trained SVM model (serialized)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 💻 Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/pratik1901-byte/Parkinson-Disease-Prediction-using-ML-DS.git
   cd Parkinson-Disease-Prediction-using-ML-DS
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in your terminal (typically `http://localhost:8501`).

---

## 📊 Using the App

### Single Patient Mode
Expand each feature group (Fundamental Frequency, Jitter, Shimmer, etc.), enter or adjust the voice-measurement values, and click **🔍 Predict** to get an instant result along with a confidence/decision score.

### Batch Mode
Upload a `.csv` file containing the required 22 feature columns (extra columns like `name` or `status` are automatically ignored). The app returns predictions for every row and lets you download the results as a CSV.

---

## 👥 Collaborators

This project was built collaboratively by:

| Name | GitHub |
|---|---|
| **Pratik Gaonkar** | [@pratik1901-byte](https://github.com/pratik1901-byte) |
| **Prajwal A B** | [@prajwal032004](https://github.com/prajwal032004) |
| **Sahana** | [@Sahanaaa9](https://github.com/Sahanaaa9) |

---

⭐ If you found this project useful, consider starring the repo!
