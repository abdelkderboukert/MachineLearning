
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost / LightGBM / CatBoost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==============================================================================
# 0. Chargement du Dataset
# ==============================================================================

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Nettoyage (TotalCharges contient des valeurs vides)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

X = df.drop(['customerID', 'Churn'], axis=1)
y = LabelEncoder().fit_transform(df['Churn'])  # 0 = No, 1 = Yes

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# ==============================================================================
# 1. Préprocessing Pipeline
# ==============================================================================

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================================
# 2. Définition des modèles
# ==============================================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=120),
    "AdaBoost": AdaBoostClassifier(n_estimators=150),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
}

# Voting & Stacking
models["Voting Classifier"] = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=500)),
        ('rf', RandomForestClassifier(n_estimators=120)),
        ('xgb', XGBClassifier(eval_metric='logloss')),
    ],
    voting='soft'
)

models["Stacking Classifier"] = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=500)),
        ('rf', RandomForestClassifier(n_estimators=120)),
    ],
    final_estimator=LogisticRegression()
)

# ==============================================================================
# 3. Entraînement + Évaluation
# ==============================================================================

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Sensitivity (Recall positive class)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    results.append([name, accuracy, auc, sensitivity, specificity])

    print("\n==============================")
    print(f"Model: {name}")
    print("==============================")
    print(classification_report(y_test, y_pred))

# ==============================================================================
# 4. Tableau comparatif des résultats
# ==============================================================================

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "ROC-AUC", "Sensitivity", "Specificity"
])
print("\n\n===== COMPARISON TABLE =====")
print(results_df)

# ==============================================================================
# 5. Courbe ROC pour le meilleur modèle
# ==============================================================================

best_model_name = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
print("\nBest model is:", best_model_name)

best_model = models[best_model_name]
pipeline_best = Pipeline([
    ('preprocess', preprocessor),
    ('model', best_model)
])
pipeline_best.fit(X_train, y_train)
y_best_prob = pipeline_best.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_best_prob)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr)
plt.title(f"ROC Curve – {best_model_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.show()

