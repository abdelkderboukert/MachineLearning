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

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", path)

# 1. Chargement des données
# REMPLACER 'Telco-Customer-Churn.csv' par le chemin de votre fichier
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset chargé avec succès. Premières lignes :")
    print(df.head())
except FileNotFoundError:
    print("Erreur: Le fichier 'Telco-Customer-Churn.csv' n'a pas été trouvé. Assurez-vous d'avoir le bon chemin.")
    # Sortir du script si le fichier n'est pas trouvé
    exit()

# 2. Nettoyage et Préparation des données

# a. Gérer les valeurs manquantes dans TotalCharges
# 'TotalCharges' est stocké comme une chaîne de caractères (object) et contient des espaces vides.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Remplacer les valeurs manquantes (NaN) par la médiane
median_totalcharges = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_totalcharges, inplace=True)

# b. Définir les variables cibles et caractéristiques
# La colonne 'customerID' n'est pas utile pour le modèle et doit être retirée
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'] # Variable cible (Oui/Non)

# c. Encodage de la variable cible (Oui/Non -> 1/0)
le = LabelEncoder()
y_encoded = le.fit_transform(y) 
# 'Yes' devient 1 (Churn), 'No' devient 0 (No Churn)

# d. Séparation des caractéristiques par type
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# 3. Création du pipeline de Prétraitement

# Créer des transformateurs pour chaque type de colonne
# Transformer pour les variables numériques (Mise à l'échelle)
numerical_transformer = StandardScaler()

# Transformer pour les variables catégorielles (One-Hot Encoding)
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

# Combiner les transformateurs dans un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# 4. Création du Pipeline ML et Entraînement du modèle

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
# random_state=42 assure la reproductibilité
# stratify=y_encoded assure la même proportion de 'Churn' dans les deux ensembles

# Définir le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# class_weight='balanced' est utile pour les datasets de churn souvent déséquilibrés

# Créer le pipeline: Prétraitement -> Modèle
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Entraîner le pipeline sur les données d'entraînement
print("\nEntraînement du modèle Random Forest...")
pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# 5. Évaluation du Modèle

# Faire des prédictions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluation de la performance
print("\n--- Évaluation du Modèle ---")
print(f"Précision (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Matrice de Confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prédit: Non Churn (0)', 'Prédit: Churn (1)'], 
            yticklabels=['Actuel: Non Churn (0)', 'Actuel: Churn (1)'])
plt.title('Matrice de Confusion ')
plt.ylabel('Valeur Réelle')
plt.xlabel('Valeur Prédite')
plt.show()

# Interprétation de la Feature Importance (en option)
# Récupérer les noms des caractéristiques après le One-Hot Encoding
feature_names = list(preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features))
cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
all_feature_names = numerical_features.tolist() + cat_feature_names

# Récupérer l'importance des caractéristiques du modèle Random Forest
feature_importances = pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 des Caractéristiques les plus Importantes :")
print(importance_df.head(10))