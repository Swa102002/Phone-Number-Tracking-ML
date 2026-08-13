import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load and clean dataset
df = pd.read_csv("phone_prefix_database.csv", dtype={'NDC': str})
df.dropna(subset=["NDC", "Cellular Operator"], inplace=True)

# Convert NDC to 4-digit and split into individual digits
df["NDC"] = df["NDC"].str.zfill(4)
features = df["NDC"].apply(lambda x: [int(d) for d in x]).tolist()
labels = df["Cellular Operator"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# ---------------------------
# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
joblib.dump(dt_model, "operator_dt_model.pkl")

# ---------------------------
# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
joblib.dump(rf_model, "operator_rf_model.pkl")

# ---------------------------
# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
joblib.dump(lr_model, "operator_lr_model.pkl")

# ---------------------------
# Print summary
print("✅ Models trained and saved:")
print(f"📊 Decision Tree Accuracy:        {dt_acc:.4f}")
print(f"🌲 Random Forest Accuracy:        {rf_acc:.4f}")
print(f"📈 Logistic Regression Accuracy:  {lr_acc:.4f}")
