import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os

# Create directories if they don't exist
os.makedirs('classification/models', exist_ok=True)
os.makedirs('classification/scalers', exist_ok=True)
os.makedirs('classification/reports', exist_ok=True)
os.makedirs('classification/results', exist_ok=True)
os.makedirs('classification/results/distributions', exist_ok=True)

# Load data
df = pd.read_csv('EDA/cardio_data.csv')

# Feature Engineering
df['age_years'] = (df['age'] / 365).round().astype(int)
df['bmi'] = df['weight'] / (df['height']/100)**2

# Define features and target
X = df[['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = df['cardio']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for continuous and categorical features
continuous_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

continuous_transformer = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline with SMOTE and Gaussian Naive Bayes
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', GaussianNB())
])

# --- Hyperparameter Tuning using GridSearchCV ---
# Define the parameter grid to search
param_grid = {
    'model__var_smoothing': np.logspace(-9, -2, num=100)
}

# Create the GridSearchCV object
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

# Train the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best pipeline
best_pipeline = grid_search.best_estimator_

# --- Save and Evaluate the Best Model ---
# Save the best model and preprocessor
joblib.dump(best_pipeline.named_steps['model'], 'classification/models/naive_bayes_model.joblib')
joblib.dump(best_pipeline.named_steps['preprocessor'], 'classification/scalers/naive_bayes_preprocessor.joblib')

# Make predictions with the best model
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

# Generate and save classification report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

report_content = f"""Best Parameters: {grid_search.best_params_}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1-Score: {f1}

Classification Report:
{report}
"""

with open('classification/reports/naive_bayes_report.txt', 'w') as f:
    f.write(report_content)

# --- Generate and Save Plots ---

# Feature Distribution plots
X_with_target = X_train.copy()
X_with_target['cardio'] = y_train
for feature in continuous_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=X_with_target, x=feature, hue='cardio', kde=True, palette='viridis', element='step')
    plt.title(f'Distribution of {feature.replace("_", " ").title()} by Cardiovascular Disease')
    plt.savefig(f'classification/results/distributions/dist_{feature}.png')
    plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('classification/results/naive_bayes_confusion_matrix.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')
plt.legend(loc='lower right')
plt.savefig('classification/results/naive_bayes_roc_curve.png')
plt.close()

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, lw=2, color='blue', label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Naive Bayes Precision-Recall Curve')
plt.legend(loc='best')
plt.savefig('classification/results/naive_bayes_pr_curve.png')
plt.close()

print("Naive Bayes training complete. Artifacts saved.")
