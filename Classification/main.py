import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ==============================================================================
# 0. Initialisation et Chargement des Données
# ==============================================================================

path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", path)

try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset chargé avec succès. Premières lignes :")
    print(df.head())
except FileNotFoundError:
    print("Erreur: Le fichier 'Telco-Customer-Churn.csv' n'a pas été trouvé. Assurez-vous d'avoir le bon chemin.")
    exit()

# ==============================================================================
# 1. Nettoyage et Préparation des données
# ==============================================================================

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_totalcharges = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_totalcharges, inplace=True)

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

le = LabelEncoder()
y_encoded = le.fit_transform(y) 

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# ==============================================================================
# 2. Création du pipeline de Prétraitement
# ==============================================================================

numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# ==============================================================================
# 4. Entraînement du modèle
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Définir le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

print("\nEntraînement du modèle Random Forest...")
pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# ==============================================================================
# 5. Fonction d'Inférence (Prédiction pour un nouveau client)
# ==============================================================================

y_pred = pipeline.predict(X_test)

print("\n--- Évaluation du Modèle ---")
print(f"Précision (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prédit: Non Churn (0)', 'Prédit: Churn (1)'], 
            yticklabels=['Actuel: Non Churn (0)', 'Actuel: Churn (1)'])
plt.title('Matrice de Confusion ')
plt.ylabel('Valeur Réelle')
plt.xlabel('Valeur Prédite')
plt.show()

feature_names = list(preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features))
cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
all_feature_names = numerical_features.tolist() + cat_feature_names

feature_importances = pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 des Caractéristiques les plus Importantes :")
print(importance_df.head(10))