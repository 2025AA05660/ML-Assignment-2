import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create models folder

os.makedirs("models", exist_ok=True)

# Load dataset

df = pd.read_csv("phishing.csv")
df.columns = df.columns.str.strip()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.replace(-1, 0)

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Models

log_reg_model = LogisticRegression(max_iter=2000)
log_reg_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

# Save Models

joblib.dump(log_reg_model, "models/logistic.pkl")
joblib.dump(dt_model, "models/decision_tree.pkl")
joblib.dump(knn_model, "models/knn.pkl")
joblib.dump(nb_model, "models/naive_bayes.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(xgb_model, "models/xgboost.pkl")

print("All models trained and saved.")
