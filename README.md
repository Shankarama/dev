# 🩺 Diabetes Risk Prediction using Neural Network

A machine learning project that predicts diabetes risk using a Multilayer Perceptron (MLP) classifier trained on patient health data. Built and run in Google Colab.

---

## 📌 Overview

This project takes a patient dataset (CSV format), trains a neural network on key health indicators, and predicts whether a patient is **at risk** or **not at risk** for diabetes — along with probability scores for each outcome.

---

## 🚀 Features

- Uploads and processes a CSV dataset directly in Google Colab
- Automatically handles missing values and encodes categorical columns
- Scales features using `StandardScaler` for optimal neural network performance
- Trains an MLP classifier with two hidden layers `(16, 8)`
- Evaluates model with accuracy score and confusion matrix heatmap
- Accepts live user input for real-time patient risk prediction

---

## 🧬 Input Features

The model expects the following patient health metrics:

| Feature | Description |
|---|---|
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age in years |

> **Target column:** `Outcome` — `1` = at risk, `0` = not at risk

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, model training, evaluation |
| `matplotlib` + `seaborn` | Confusion matrix visualization |
| `google.colab` | File upload in Colab environment |

---

## ▶️ How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Run all cells
3. Upload your CSV file when prompted (must include an `Outcome` column)
4. After training, enter patient values when prompted for a live prediction

```
Glucose: 120
BloodPressure: 70
Insulin: 85
BMI: 28.5
DiabetesPedigreeFunction: 0.45
Age: 35
```

**Sample output:**
```
Prediction for the input data: not at risk
Probability of not at risk (Class 0): 0.8231
Probability of at risk (Class 1): 0.1769
```

---

## 📊 Model Architecture

```
Input Layer  →  Hidden Layer (16 neurons)  →  Hidden Layer (8 neurons)  →  Output Layer
```

- **Optimizer:** Adam (default in `MLPClassifier`)
- **Max iterations:** 500
- **Train/Test split:** 80% / 20%

---

## 📁 Dataset

The project expects a CSV file with the following columns (at minimum):

```
Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
```

A commonly used compatible dataset is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) available on Kaggle.

---

## 📈 Evaluation Metrics

After training, the model outputs:
- **Accuracy score** on the test set
- **Confusion matrix** heatmap (Actual vs. Predicted)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
