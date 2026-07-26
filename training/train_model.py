import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

DATA_PATH = "dataset/student_study_data.csv"
MODEL_DIR = "model"

df = pd.read_csv(DATA_PATH)

# Label: merge "Advanced" into "Intermediate" -- the original 3-class split had only
# 9/500 rows labeled Advanced, too few to reliably model as a separate class.
df['course_category'] = np.where(df['final_grade'] > 60, 'Intermediate', 'Beginner')
le_course = LabelEncoder()
df['course_encoded'] = le_course.fit_transform(df['course_category'])
print("Label distribution:\n", df['course_category'].value_counts(), "\n")

# Features: behavioral only. final_grade is deliberately EXCLUDED -- it was used
# to build the label above, so including it as a feature would leak the answer
# directly into the model (this was a bug in the original version).
features = ['study_hours_per_week', 'attendance_percentage', 'sleep_hours_per_day',
            'assignments_completed', 'extracurricular_Yes', 'part_time_job_Yes',
            'internet_access_Yes']
X = df[features]
y = df['course_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning for the regularization strength C
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    {'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='f1_macro'
)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best C:", grid.best_params_)

# 5-fold cross-validation for an honest, stable accuracy estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

y_pred = model.predict(X_test)
print("\nHold-out test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_course.classes_))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le_course, os.path.join(MODEL_DIR, "course_encoder.pkl"))
joblib.dump(features, os.path.join(MODEL_DIR, "feature_order.pkl"))
print("\nSaved model, scaler, course_encoder, feature_order to", MODEL_DIR)
