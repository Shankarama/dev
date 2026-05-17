# Diabetes Risk Prediction using Neural Network (MLP Classifier)

> A Machine Learning project that predicts whether a patient is at risk of diabetes based on key health indicators, using a Multilayer Perceptron (MLP) Neural Network — built and deployed in Google Colab.

---

## Problem Statement

Diabetes is one of the most prevalent chronic diseases worldwide, affecting millions of people and leading to serious health complications if left undetected. Early diagnosis is critical for effective treatment and management.

Manual diagnosis requires extensive medical tests and expert interpretation, which may not always be accessible. The goal of this project is to:

- Build an intelligent system that can **predict diabetes risk** based on basic patient health data
- Provide **probability scores** for at-risk and not-at-risk classifications
- Enable **real-time prediction** by accepting live user input

> By applying machine learning to healthcare data, this project aims to assist in early detection of diabetes, potentially saving lives through timely intervention.

---

## Dataset

- **Format:** CSV (uploaded manually in Google Colab)
- **Target Column:** `Outcome` — `1` = Diabetic (at risk), `0` = Non-Diabetic (not at risk)
- **Commonly Used Dataset:** [Pima Indians Diabetes Database — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Dataset Columns

| Column | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | Target variable (0 = Not at risk, 1 = At risk) |

---

## Project Flow

```
Raw CSV Dataset
│
▼
Data Loading & Exploration
│
▼
Data Preprocessing
(Handle missing values, Label Encoding, Feature Scaling)
│
▼
Train / Test Split (80% / 20%)
│
▼
Model Training (MLP Neural Network)
│
▼
Model Evaluation
(Accuracy, Confusion Matrix)
│
▼
Live User Input
│
▼
Real-Time Prediction + Probability Score
```

---

## Data Preprocessing

Before training the model, the following preprocessing steps were applied:

1. **Loading the Dataset**
- Dataset is uploaded via Google Colab's `files.upload()` and loaded into a Pandas DataFrame.

2. **Exploratory Data Analysis**
- Checked dataset shape and first few rows using `df.head()`
- Identified missing or null values using `df.isnull().sum()`

3. **Label Encoding**
- Any categorical (object-type) columns are automatically detected and converted to numeric values using `LabelEncoder` from scikit-learn.

4. **Feature & Target Separation**
- Features (`X`): All columns except `Outcome`
- Target (`y`): The `Outcome` column

5. **Feature Scaling**
- Applied `StandardScaler` to normalize all feature values to a mean of 0 and standard deviation of 1.
- Ensures equal contribution of all features during neural network training.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## ️ Train Test Split

The dataset is split into training and testing sets using an **80/20 ratio**:

```python
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42
)
```

| Split | Percentage | Purpose |
|---|---|---|
| Training Set | 80% | Used to train the MLP model |
| Testing Set | 20% | Used to evaluate model performance |

- `random_state=42` ensures reproducibility of results.

---

## Algorithm Used

### Multilayer Perceptron (MLP) — Neural Network

An MLP is a type of **feedforward artificial neural network** that consists of multiple layers of neurons. It learns complex patterns in data through **backpropagation** and **gradient descent**.

```python
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
```

### Architecture

```
Input Layer (8 features)
│
Hidden Layer 1 — 16 Neurons (ReLU Activation)
│
Hidden Layer 2 — 8 Neurons (ReLU Activation)
│
Output Layer — 1 Neuron (Sigmoid → Binary Classification)
```

| Parameter | Value |
|---|---|
| Hidden Layers | 2 |
| Neurons per Layer | 16, 8 |
| Activation Function | ReLU (default) |
| Optimizer | Adam (default) |
| Max Iterations | 500 |
| Random State | 42 |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, model training, and evaluation |
| `matplotlib` | Plotting and data visualization |
| `seaborn` | Confusion matrix heatmap rendering |
| `google.colab` | File upload support in Colab environment |
| `io` | Reading uploaded file bytes |

Install required libraries locally:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Evaluation Metrics

The model is evaluated using the following metrics on the test set:

| Metric | Description |
|---|---|
| **Accuracy** | Percentage of correctly classified samples |
| **Confusion Matrix** | Table showing TP, TN, FP, FN prediction counts |

```python
print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

### Confusion Matrix Layout

```
Predicted: 0 Predicted: 1
Actual: 0 [ True Negative False Positive ]
Actual: 1 [ False Negative True Positive ]
```

| Term | Meaning |
|---|---|
| **True Positive (TP)** | At-risk patients correctly identified |
| **True Negative (TN)** | Healthy patients correctly identified |
| **False Positive (FP)** | Healthy patients incorrectly flagged as at-risk |
| **False Negative (FN)** | At-risk patients missed by the model |

---

## Result

After training the MLP classifier on the Pima Indians Diabetes Dataset:

- The model achieves a strong **accuracy score** on the unseen 20% test split
- The **confusion matrix heatmap** provides a clear visual breakdown of predictions
- Class probabilities give a **confidence score** for each prediction made
- The model reliably distinguishes between diabetic and non-diabetic patients

> The neural network successfully learns patterns from health indicators to classify diabetes risk with meaningful accuracy.

---

## Example Prediction

After training, the model accepts live patient data for real-time prediction:

**Sample Input:**

```
Glucose: 148
BloodPressure: 72
Insulin: 0
BMI: 33.6
DiabetesPedigreeFunction: 0.627
Age: 50
```

**Sample Output:**

```
Prediction for the input data: at risk
Probability of not at risk (Class 0): 0.2134
Probability of at risk (Class 1): 0.7866
```

> The model predicts the patient is **at risk** for diabetes with **78.66% confidence**.

---

## ▶️ How to Run the Project

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `.ipynb` notebook file
3. Run all cells (`Runtime → Run All`)
4. When prompted, **upload your CSV dataset**
5. After training completes, **enter patient values** for a live prediction

### Option 2: Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/your-username/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

2. **Install dependencies**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Update the file loading section** — replace the Colab upload block with:

```python
df = pd.read_csv('your_dataset.csv')
```

4. **Run the script**

```bash
python diabetes_prediction.py
```

---

## Future Improvements

- **Additional Features** — Include HbA1c, cholesterol levels, and physical activity data for better accuracy
- **Model Comparison** — Benchmark MLP against Random Forest, XGBoost, and SVM classifiers
- **Web Deployment** — Deploy as a web application using **Flask** or **Streamlit** for public access
- **Mobile App** — Integrate the model into a mobile health app for on-the-go predictions
- **Cross-Validation** — Apply k-fold cross-validation for more robust model evaluation
- **Hyperparameter Tuning** — Use `GridSearchCV` to optimize MLP parameters
- ️ **Larger Dataset** — Train on a more diverse dataset to improve generalization
- **Extended Metrics** — Add Precision, Recall, F1-Score, and ROC-AUC curve for deeper evaluation

---

## ‍ Team Details

This project was developed by the following students as part of an academic Machine Learning initiative:

| Name | USN | Role |
|---|---|---|
| **Shrikant** | 4MC25CS286 | Model Development & Training |
| **Samprit** | 4MC25CS256 | Data Preprocessing & Analysis |
| **Shankaramurti** | 4MC25CS272 | Evaluation & Visualization |
| **Sachin** | 4MC25CS252 | Documentation & Testing |

> Department of Computer Science & Engineering

---

## Conclusion

This project successfully demonstrates the application of a **Multilayer Perceptron (MLP) Neural Network** for early diabetes risk prediction using patient health indicators.

### Key Achievements

- ️ Built a complete end-to-end ML pipeline from raw data to real-time prediction
- ️ Applied effective preprocessing — label encoding, null-value checks, and standard scaling
- ️ Trained a two-layer neural network `(16, 8)` achieving reliable classification accuracy
- ️ Visualized model performance using a confusion matrix heatmap
- ️ Enabled live patient risk assessment through interactive user input

### Learnings

Through this project, the team gained hands-on experience in:
- Building and evaluating neural network classifiers using scikit-learn
- Understanding the role of data preprocessing in model performance
- Interpreting classification metrics and confusion matrices
- Applying machine learning to solve real-world healthcare problems

### Impact

Early detection of diabetes can significantly improve patient outcomes by enabling timely medical intervention. This project lays the foundation for a practical, AI-powered diagnostic support tool that can be further developed into a deployable healthcare application.

> *"Machine learning in healthcare is not about replacing doctors — it's about empowering them with better tools for early and accurate diagnosis."*

---

## License

This project is open-source and available under the [MIT License](LICENSE).
