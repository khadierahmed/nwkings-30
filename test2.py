import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('fit_training_data.csv')  # Make sure this file is present

# Optional: Check data quality
print("Sample data:\n", df.head())
print("Class distribution:\n", df['fit_label'].value_counts())

# Feature columns and target
X = df[['semantic_similarity', 'keyword_overlap', 'skills_matched', 'resume_length']]
y = df['fit_label']  # Assumes 0 = Poor, 1 = Average, 2 = Good

# Split data for evaluation (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'fit_predictor.pkl')
print("✅ Model saved as 'fit_predictor.pkl'")
