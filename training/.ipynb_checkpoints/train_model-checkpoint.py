from queue import Full


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "60a6df0f-77d3-432d-a342-7194990c3f39",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder,StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.ensemble import VotingClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "27c7d1f4-94f6-4760-b7a2-cd0720318525",
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_PATH = \"../dataset/student_study_data.csv\"\n",
    "MODEL_DIR = \"model\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "61762432-b05d-4062-ab2f-852fd51bfb32",
   "metadata": {},
   "outputs": [],
   "source": [
    "if not os.path.exists(DATA_PATH):\n",
    "    raise FileNotFoundError(\"Please add your dataset to 'dataset/student_study_data.csv'.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "542b8815-78f1-4dc1-b90d-a44ff3dee73a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset loaded with 500 rows\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(DATA_PATH)\n",
    "print(f\"Dataset loaded with {len(df)} rows\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "0a67ab7b-05b2-46bd-aa92-4b23d9494331",
   "metadata": {},
   "outputs": [],
   "source": [
    "required = ['study_hours_per_week', 'attendance_percentage', 'final_grade', 'extracurricular_Yes', 'part_time_job_Yes']\n",
    "for col in required:\n",
    "    if col not in df.columns:\n",
    "        raise ValueError(f\"Column '{col}' not found in dataset. Please rename your dataset headers.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "0f55687c-b204-40ca-b4f2-7316d9077dcc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna(subset=required)\n",
    "df = df.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "ecc65002-4a74-4030-8137-e7cda4abd1ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'interest_area' in df.columns:\n",
    "    le_interest = LabelEncoder()\n",
    "    df['interest_area_encoded'] = le_interest.fit_transform(df['interest_area'].astype(str))\n",
    "else:\n",
    "    le_interest = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "9bccda40-8de7-419b-b230-cb84fa3e2595",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['course_category'] = np.where(df['final_grade'] > 80, 'Advanced',\n",
    "                                 np.where(df['final_grade'] > 60, 'Intermediate', 'Beginner'))\n",
    "\n",
    "le_course = LabelEncoder()\n",
    "df['course_encoded'] = le_course.fit_transform(df['course_category'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "5476c7c6-f947-4585-a46a-2a56d8757938",
   "metadata": {},
   "outputs": [],
   "source": [
    " features = required.copy()\n",
    "if le_interest:\n",
    "    features.append('interest_area_encoded')\n",
    "X = df[features]\n",
    "y = df['course_encoded']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "62860346-fd19-4054-964b-b308b95c1656",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "11cb5a71-7482-444d-bf53-555c3c61f98f",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "e03bfd67-f8cb-44ed-a0a6-3b9b3fbbfcd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LogisticRegression(max_iter=1000)\n",
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "nb = GaussianNB()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "575f12f1-27e4-46eb-a17d-d5f3585b32e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "ensemble = VotingClassifier(estimators=[\n",
    "    ('lr', lr),\n",
    "    ('knn', knn),\n",
    "    ('nb', nb)\n",
    "], voting='hard')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "2a4ec8e6-8008-4933-9413-795418517a58",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training ensemble...\n",
      "Training complete!\n",
      "Accuracy: 0.96\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "    Advanced       1.00      0.50      0.67         2\n",
      "    Beginner       0.94      0.97      0.95        32\n",
      "Intermediate       0.97      0.97      0.97        66\n",
      "\n",
      "    accuracy                           0.96       100\n",
      "   macro avg       0.97      0.81      0.86       100\n",
      "weighted avg       0.96      0.96      0.96       100\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Training ensemble...\")\n",
    "ensemble.fit(X_train, y_train)\n",
    "y_pred = ensemble.predict(X_test)\n",
    "print(\"Training complete!\")\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred, target_names=le_course.classes_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "e91e13a5-345f-416c-bcc1-b5807626e99d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['model\\\\course_encoder.pkl']"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.makedirs(MODEL_DIR, exist_ok=True)\n",
    "joblib.dump(ensemble, os.path.join(MODEL_DIR, \"model.pkl\"))\n",
    "joblib.dump(scaler, os.path.join(MODEL_DIR, \"scaler.pkl\"))\n",
    "joblib.dump(le_course, os.path.join(MODEL_DIR, \"course_encoder.pkl\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "0d33e3ed-931c-4bab-a235-81ea6233ca39",
   "metadata": {},
   "outputs": [],
   "source": [
    "if le_interest:\n",
    "    joblib.dump(le_interest, os.path.join(MODEL_DIR, \"interest_encoder.pkl\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": Full,
   "id": "07ed3fef-3978-434a-820c-1f79c8b74de5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "All model files saved successfully in 'model/' folder\n"
     ]
    }
   ],
   "source": [
    "print(\"All model files saved successfully in 'model/' folder\") "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (tf-env)",
   "language": "python",
   "name": "tf-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
