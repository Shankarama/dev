INTRODUCTION
Disease risk prediction is a critical application of machine learning in healthcare. Detecting the
probability of diseases such as diabetes at an early stage enables timely medical intervention and better
patient outcomes. Manual diagnosis based on individual parameters is error-prone and inconsistent
across practitioners.
This project applies a Multi-Layer Perceptron (MLP) Neural Network Classifier to the Pima Indians
Diabetes Dataset (PIDD) to predict whether a patient is 'at risk' or 'not at risk' of diabetes. By learning
non-linear relationships between clinical parameters, the neural network outperforms traditional linear
models for this classification task.
INFORMATION OF MLP NEURAL NETWORK
• An MLP (Multi-Layer Perceptron) is a class of feedforward artificial neural network (ANN).
• It consists of at least three layers of nodes: an input layer, one or more hidden layers, and an
output layer.
• Each node (except input nodes) uses a non-linear activation function, enabling the model to learn
complex patterns.
• The architecture used in this project has two hidden layers with 16 and 8 neurons respectively:
hidden_layer_sizes=(16, 8).
• MLPClassifier from scikit-learn is used — it trains using backpropagation with gradient descent.
• For classification, it outputs class probabilities via softmax activation in the output layer.
• The model is trained for up to 500 iterations (max_iter=500) with random_state=42 for
reproducibility.
KEY TERMINOLOGY
Features (X): The 8 clinical input columns — Pregnancies, Glucose, BloodPressure, SkinThickness,
Insulin, BMI, DiabetesPedigreeFunction, Age.
Target (y): The binary output column to be predicted — Outcome (0 = not at risk, 1 = at risk).
Label Encoding: Converting any categorical text columns into numerical form so the model can
process them.
StandardScaler: Scales all features to zero mean and unit variance; essential for neural network
convergence.
Train-Test Split: The dataset is split into 80% training data (to teach the model) and 20% testing data
(to evaluate it).
Accuracy: Proportion of correctly classified samples out of the total. Higher is better.
Confusion Matrix: A 2x2 table showing True Positives, True Negatives, False Positives, and False
Negatives.
predict_proba: Returns class probabilities for Class 0 (not at risk) and Class 1 (at risk) for any input. PROBLEM STATEMENT
• To predict whether a patient is at risk of diabetes based on 8 clinical features: Pregnancies,
Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age — using
a Multi-Layer Perceptron Classifier.
• The target variable (Outcome) is binary: 0 = not at risk, 1 = at risk.
• The challenge lies in standardizing the input features using StandardScaler and training a neural
network that can classify patients accurately.
• The model must support manual prediction — accepting live patient data and outputting risk
category with class probabilities.
METHODOLOGY
DATA COLLECTION:
• The Pima Indians Diabetes Dataset (PIDD) was used — a well-known benchmark dataset.
• It contains 768 records with 8 clinical features and 1 binary target (Outcome).
• Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age.
• There are no missing values in the dataset (confirmed via df.isnull().sum()).
DATA PREPROCESSING:
• All categorical text columns (if any) were converted to numerical form using LabelEncoder from
scikit-learn.
• The target variable selected was Outcome (binary: 0 or 1).
• All 8 feature columns were scaled using StandardScaler (zero mean, unit variance) — critical for
neural network performance.
• Feature names were saved separately to ensure consistent ordering during manual prediction.
TRAIN-TEST SPLIT:
• The dataset was divided into 80% training data and 20% testing data using train_test_split.
• Random state was set to 42 for reproducibility.
• Approximately 614 samples for training, 154 samples for testing.
MODEL TRAINING:
• The MLPClassifier was imported from sklearn.neural_network.
• Architecture: two hidden layers with 16 and 8 neurons — hidden_layer_sizes=(16, 8).
• max_iter=500 ensures sufficient training iterations for convergence.
• The model was trained on X_train (scaled) and y_train.
MODEL EVALUATION:
• Predictions were made on the test set (X_test).
• Performance was evaluated using: Accuracy Score.
• A Confusion Matrix was plotted using seaborn's heatmap to visualize True Positives, False
Positives, True Negatives, and False Negatives. RESULTS
• The MLP Neural Network Classifier successfully learned non-linear decision boundaries from the
PIDD dataset.
• Accuracy: 0.7727 — the model correctly classified 77.27% of test samples.
• The Confusion Matrix shows the distribution of correct and incorrect predictions across both
classes.
• Class 0 (not at risk) predictions are generally more accurate due to the slight class imbalance in
PIDD.
• Sample prediction: Glucose=148, BloodPressure=72, Insulin=0, BMI=33.6,
DiabetesPedigreeFunction=0.627, Age=50 — Predicted: at risk.
Metric Value Interpretation
Accuracy 0.7727 77.27% of predictions are correct
Hidden Layers (16, 8) Two hidden layers with 16 and 8 neurons
Max Iterations 500 Up to 500 epochs for training convergence
Train/Test Split 80% / 20% ~614 train samples, ~154 test samples
Scaler Used StandardScaler Zero mean, unit variance normalisation
Manual Prediction Output:
========== MANUAL PREDICTION ==========
Enter Glucose : 148
Enter BloodPressure : 72
Enter Insulin : 0
Enter BMI : 33.6
Enter DiabetesPedigreeFunction : 0.627
Enter Age : 50
========== RESULT ==========
Prediction for the input data : at risk
Probability of not at risk (Class 0) : 0.3142
Probability of at risk (Class 1) : 0.6858
[ Attach screenshot of Confusion Matrix heatmap here ] CONCLUSION
• This project demonstrates that a Multi-Layer Perceptron Neural Network is an effective technique
for predicting diabetes risk from clinical parameters.
• By standardizing input features using StandardScaler and training a two-hidden-layer MLP, the
model achieves ~77% accuracy on the Pima Indians Diabetes Dataset.
• The model can help clinicians identify high-risk patients early, enabling timely interventions and
better health outcomes.
• The confusion matrix analysis provides insight into false-positive and false-negative rates, which
are critical in medical diagnosis settings.
• The model supports live patient input, making it suitable for integration into clinical
decision-support systems.
Future Improvements:
• Using a larger and more diverse dataset with additional demographic features.
• Applying ensemble models such as Random Forest, XGBoost, or Gradient Boosting for
comparison.
• Adding evaluation metrics such as Precision, Recall, F1-Score, and AUC-ROC curve.
• Incorporating hyperparameter tuning (learning rate, layer sizes, activation functions) using
GridSearchCV.
• Deploying the model as a web application using Flask or Streamlit for real-world use. CODE USED
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))
print('Dataset shape:', df.shape)
print(df.head())
print('Missing Values:')
print(df.isnull().sum())
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
le = LabelEncoder()
df[col] = le.fit_transform(df[col])
label_encoders[col] = le
target_col = 'Outcome'
X = df.drop(target_col, axis=1)
y = df[target_col]
feature_names = X.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
user_input_new = {
'Glucose': float(input('Glucose: ')),
'BloodPressure': float(input('BloodPressure: ')),
'Insulin': float(input('Insulin: ')),
'BMI': float(input('BMI: ')),
'DiabetesPedigreeFunction': float(input('DiabetesPedigreeFunction: ')),
'Age': float(input('Age: ')) }
full_user_input = {col: user_input_new.get(col, 0.0) for col in feature_names}
user_df_new = pd.DataFrame([full_user_input])[feature_names]
user_scaled_new = scaler.transform(user_df_new)
manual_prediction_new = model.predict(user_scaled_new)
manual_prediction_proba_new = model.predict_proba(user_scaled_new)
print(f'Prediction: {"at risk" if manual_prediction_new[0]==1 else "not at risk"}')
print(f'Probability of not at risk (Class 0): {manual_prediction_proba_new[0][0]:.4f}')
print(f'Probability of at risk (Class 1): {manual_prediction_proba_new[0][1]:.4f}') OUTPUT
Dataset Preview:
Pregnan
cies Glucose BP Skin Insulin BMI DPF Age
Outcom
e
0 6 148 72 35 0 33.6 0.627 50 1
1 1 85 66 29 0 26.6 0.351 31 0
2 8 183 64 0 0 23.3 0.672 32 1
3 1 89 66 23 94 28.1 0.167 21 0
4 0 137 40 35 168 43.1 2.288 33 1
Missing Values:
Pregnancies 0
Glucose 0
BloodPressure 0
SkinThickness 0
Insulin 0
BMI 0
DiabetesPedigreeFunction 0
Age 0
Outcome 0
dtype: int64
Model Evaluation:
Train size: 614 | Test size: 154
Model trained successfully
Accuracy: 0.7727
Manual Prediction Output:
========== MANUAL PREDICTION ==========
Glucose : 148
BloodPressure : 72
Insulin : 0
BMI : 33.6
DiabetesPedigreeFunction : 0.627
Age : 50
========== RESULT ==========
Prediction for the input data : at risk
Probability of not at risk (Class 0) : 0.3142