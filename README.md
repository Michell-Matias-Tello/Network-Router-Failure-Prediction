# Network Router Failure Prediction

---

## **Overview**
This project is designed to estimate the **failure probability of network routers** using operational variables. By applying **machine learning techniques**, the system predicts potential failures, enabling **proactive maintenance** and **reducing downtime** in network infrastructures. The solution leverages **supervised learning models**, feature engineering, and calibration techniques to deliver **reliable, actionable insights** for network reliability teams.

The repository provides a **comprehensive pipeline**, from data preprocessing to model evaluation, including **visualizations, performance metrics, and threshold optimization** for decision-making.

---

## **Project Structure**
```text
network-router-failure-prediction/
├── data.py                  # Data loading, preprocessing, and feature engineering
├── main.py                  # Main pipeline: model training, evaluation, and output generation
├── outputs/                 # Generated visualizations, metrics, and analysis results
│   ├── 01_distributions_professional.png
│   ├── 02_correlation_matrix_professional.png
│   ├── 03_class_imbalance_professional.png
│   ├── 04_feature_target_relationships_professional.png
│   ├── model_comparison.csv
│   ├── calibration_curve_threshold.png
│   ├── gain_lift_charts.png
│   ├── cost_vs_threshold.png
│   ├── threshold_analysis_summary.csv
│   ├── calibration_random_forest
│   ├── calibration_xgboost.png
│   ├── prob_dist_xgboost.png
│   └── prob_dist_random_forest.png
│
├── data/                    # Processed datasets and feature metadata
│   ├── X_train_scaled.npy
│   ├── X_test_scaled.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── feature_names.csv
│
├── models/                  # Serialized models and scalers
│   ├── scaler.pkl
│   ├── logistic_regression_best.pkl
│   ├── random_forest_best.pkl
│   ├── xgboost_best.pkl
│   ├── random_forest_sigmoid.pkl
│   ├── random_forest_isotonic.pkl
│   ├── xgboost_sigmoid.pkl
│   └── xgboost_isotonic.pkl
│
├── requirements.txt
└── datos_routers/
```

---

## **Objectives**
- **Predictive Modeling**: Develop **high-accuracy models** to estimate router failure probability using operational variables.
- **Feature Analysis**: Identify **key indicators** of failure through exploratory data analysis and correlation studies.
- **Model Calibration**: Improve **probability reliability** using Platt Scaling and Isotonic Regression.
- **Threshold Optimization**: Balance **false positives and negatives** to minimize operational costs.
- **Actionable Insights**: Provide **visualizations and metrics** to support data-driven maintenance strategies.

---

## **Methodology**

### **1. Data Preprocessing**
- **Feature Scaling**: Standardization of operational variables using `StandardScaler`.
- **Train-Test Split**: Stratified splitting to maintain class distribution.
- **Feature Selection**: Analysis of feature importance and redundancy.

### **2. Exploratory Data Analysis (EDA)**
- **Distributions**: Visualization of feature and target distributions.
- **Correlation Matrix**: Identification of multicollinearity and feature relationships.
- **Class Imbalance**: Assessment of failure/non-failure ratio and mitigation strategies.
- **Feature-Target Relationships**: Univariate analysis of operational variables vs. failure probability.

### **3. Model Training**
- **Logistic Regression**: Baseline linear model for binary classification.
- **Random Forest**: Ensemble method for non-linear relationships and feature importance.
- **XGBoost**: Gradient-boosted trees for high performance and robustness.

### **4. Model Calibration**
- **Platt Scaling**: Logistic regression-based calibration for sigmoid transformation.
- **Isotonic Regression**: Non-parametric calibration for monotonic probability adjustment.

### **5. Evaluation**
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- **Calibration Curves**: Comparison of predicted vs. actual probabilities.
- **Cost Analysis**: Evaluation of misclassification costs at various thresholds.

### **6. Threshold Optimization**
- **Gain/Lift Charts**: Assessment of model effectiveness in identifying high-risk routers.
- **Cost vs. Threshold**: Optimization of decision thresholds to minimize operational costs.

---

## **Models**
| Model                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Logistic Regression       | Linear model for baseline performance.                                     |
| Random Forest             | Ensemble of decision trees for non-linear patterns.                       |
| XGBoost                   | Gradient-boosted trees for high accuracy and speed.                        |
| Random Forest (Calibrated)| RF with Platt Scaling or Isotonic Regression for reliable probabilities.   |
| XGBoost (Calibrated)      | XGBoost with Platt Scaling or Isotonic Regression for reliable probabilities.|

---

## **Key Outputs**
- **Visualizations**:
  - Feature distributions, correlation matrices, class imbalance, and feature-target relationships.
  - Calibration curves (before/after calibration) for all models.
  - Probability distributions of predicted failure probabilities.
- **Metrics**:
  - Model comparison CSV with performance metrics.
  - Threshold analysis summary for cost optimization.
- **Charts**:
  - Gain/lift charts for model effectiveness.
  - Cost vs. threshold analysis for decision-making.

---

## **How to Use**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/network-router-failure-prediction.git
   cd network-router-failure-prediction
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Data**:
   - Place raw operational data in the `datos_routers/` directory.
   - Ensure the data includes operational variables and failure labels.

4. **Run the Pipeline**:
   ```bash
   python main.py
   ```
   - This script executes:
     - Data loading and preprocessing (`data.py`).
     - Model training, calibration, and evaluation.
     - Generation of all outputs in the `outputs/` directory.

5. **Review Results**:
   - Check the `outputs/` directory for:
     - Visualizations (PNG files).
     - Performance metrics (CSV files).
     - Calibration and threshold analysis.

---

## **Dependencies**
- Python 3.8+
- Key libraries:
  - `pandas`, `numpy`
  - `scikit-learn`, `xgboost`
  - `matplotlib`, `seaborn`
  - `joblib`, `pickle`

---

## **Author**
This project is developed and maintained as part of an initiative to enhance **network reliability** through **predictive analytics** and **machine learning**.
