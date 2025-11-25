# Stock-Price-Prediction-using-ML-DL-Interactive-Dashboard
# Stock Forecaster: End-to-End Time Series Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-yellow)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

##  Project Overview

Predicting financial market trends is a complex challenge due to market volatility and non-linear patterns. This project builds a robust **Time-Series Forecasting Engine** that predicts the closing price of the **NIFTY 50** index. 

Instead of relying on a single algorithm, this system implements a **multi-model benchmarking pipeline**, training over **15 different algorithms** (Classical ML, Gradient Boosting, and Deep Learning) across varying look-back windows (30, 60, 90, 120 days) to identify the optimal predictor. The best-performing models are deployed via an interactive **Streamlit Dashboard** connected to live market data.

---

##  Dashboard Demo

<img width="1366" height="623" alt="Screenshot (1062)" src="https://github.com/user-attachments/assets/c0ac3671-2efd-496b-a415-4c87ff9a2a1c" />

> *The interactive dashboard allows users to toggle between models, perform historical validation, and forecast future prices using live data via the Yahoo Finance API.*

---

##  Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Deep Learning:** TensorFlow, Keras (RNN, LSTM, GRU, Bi-LSTM)
* **Visualization:** Plotly (Interactive), Matplotlib
* **Deployment:** Streamlit
* **Data Source:** Yahoo Finance (`yfinance`)

---

##  Methodology & Modeling Pipeline

### 1. Feature Engineering (Sliding Window)
To transform the time-series data into a supervised learning format, the system generates datasets using a sliding window approach.
* **Input:** $X$ days of historical prices (Open, Close, High, Low).
* **Target:** Price on Day $X+1$.
* **Variations:** Windows of 30, 60, 90, and 120 days were tested for every model.

### 2. Algorithms Evaluated
The project benchmarks a wide "Model Zoo" to capture different data characteristics:
* **Linear Models:** Linear Regression, Ridge, Lasso.
* **Ensemble Methods:** Random Forest, Gradient Boosting.
* **State-of-the-Art Boosting:** XGBoost, LightGBM.
* **Distance-Based:** K-Nearest Neighbors (KNN).
* **Deep Learning:**
    * **RNN:** Simple Recurrent Neural Networks.
    * **LSTM:** Long Short-Term Memory networks (great for long-term dependencies).
    * **GRU:** Gated Recurrent Units.
    * **Bi-LSTM:** Bidirectional LSTMs.

### 3. Evaluation
Models were evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**. The pipeline automatically serializes (saves) the highest-performing models for deployment.

---

##  Installation & Usage

### Prerequisites
* Python 3.8+
* Pip

### Setup
1.  **Clone the repository:**
    ```bash
   git clone https://github.com/Paarija/Stock-Price-Prediction-using-ML-DL-Interactive-Dashboard.git
   cd Stock-Price-Prediction-using-ML-DL-Interactive-Dashboard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Models (Optional):**
    * If `trained_models.joblib` is missing, run the `stock pred.ipynb` notebook.
    * This will process `nifty50data.csv`, train all models, and save the artifacts.

4.  **Run the Application:**
    ```bash
    streamlit run forecasting_dashboard.py
    ```

---

##  Key Insights
* **Window Importance:** The 90-day look-back window often provided the best balance between capturing trends and avoiding noise for this specific dataset.
* **Model Performance:** While LSTMs are powerful, distance-based algorithms like **KNN** and ensemble methods like **Random Forest** showed surprising resilience and accuracy in short-term volatility prediction compared to complex deep learning architectures on this specific timeframe.

---
