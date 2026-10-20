from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "synthetic_heat_worker_data.csv")
target = "simulated_heat_strain_label"
X, y = df.drop(columns=[target]), df[target]
cat = ["worker_type", "sensor_quality"]
num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
    ("numeric", "passthrough", num)
])
model = Pipeline([
    ("preprocess", pre),
    ("classifier", RandomForestClassifier(n_estimators=160, max_depth=9,
        min_samples_leaf=4, class_weight="balanced", random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Synthetic-label evaluation only — not real-world/clinical performance")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred, zero_division=0))
(ROOT / "model").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "model" / "thermotwin_model.joblib")
print("Saved model/model file.")
