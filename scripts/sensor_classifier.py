import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from datetime import timedelta, datetime
from sklearn.model_selection import cross_val_score, train_test_split

from helpers import get_or_create_experiment

# Simulation settings
sampling_rate = 10  # Hz
total_hours = 5
total_seconds = total_hours * 3600
total_samples = total_seconds * sampling_rate
n_estimators=100

# Time index
start_time = datetime.now()
timestamps = [
  start_time + timedelta(seconds=i / sampling_rate) for i in range(total_samples)
]

# Simulate sensor readings
np.random.seed(42)
sns.set_theme()

# Using set_style
sns.set_style("darkgrid")


# Force sensors: normal range 0–5, collision spike: 20–30
def simulate_force_data(length, spike_indices):
  base = np.random.normal(loc=2.5, scale=0.5, size=length)
  for idx in spike_indices:
      base[idx : idx + 5] += np.random.uniform(20, 30)  # simulate short spike
  return base


# Accelerometers: normal vibration range -1 to 1, spike ±5
def simulate_accel_data(length, spike_indices):
  base = np.random.normal(loc=0, scale=0.5, size=length)
  for idx in spike_indices:
      base[idx : idx + 5] += np.random.uniform(-5, 5)
  return base


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
experiment_id = get_or_create_experiment("Sensor Collision Classifier")

mlflow.set_experiment(experiment_id=experiment_id)

# Randomly choose true collision timestamps (e.g., 20 events)
true_collision_indices = sorted(random.sample(range(1000, total_samples - 1000), 20))

# Randomly choose false positive (vibration) events (e.g., 100 events)
false_positive_indices = random.sample(
  sorted(set(range(1000, total_samples - 1000)) - set(true_collision_indices)), 100
)

# Generate signals
data = {
  "timestamp": timestamps,
  "force1": simulate_force_data(
    total_samples, true_collision_indices + false_positive_indices
  ),
  "force2": simulate_force_data(
    total_samples, true_collision_indices + false_positive_indices
  ),
  "force3": simulate_force_data(total_samples, true_collision_indices),
  "force4": simulate_force_data(total_samples, false_positive_indices),
  "accel1": simulate_accel_data(total_samples, false_positive_indices),
  "accel2": simulate_accel_data(total_samples, true_collision_indices),
}

# Create label array
labels = np.zeros(total_samples, dtype=int)
for idx in true_collision_indices:
    labels[idx : idx + 5] = 1  # label the spike as a collision

data["label"] = labels
run_name = "first_attempt"

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("synthetic_train_sensor_data.csv", index=False)
print("✅ Synthetic dataset saved as 'synthetic_train_sensor_data.csv'")

X = df.drop(["timestamp", "label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
classification_scores = classification_report(y_test, y_pred, output_dict=True)
scores = cross_val_score(clf, X, y, cv=5, scoring="f1")

importances = clf.feature_importances_
feat_names = X.columns


# Plot
def plot_feature_importances(importances, feature_names, plot_size=(16, 12)):
  fig, ax = plt.subplots(figsize=plot_size)
  sns.barplot(x=importances, y=feature_names, ax=ax)
  plt.title("Feature Importances")
  plt.xlabel("Importance", fontsize=12)
  plt.ylabel("Feature", fontsize=12)

  # Add legend to explain the lines
  ax.legend()
  plt.tight_layout()
  plt.close(fig)
  return fig


def plot_confusion_matrix(y_test, y_pred, plot_size=(16, 12)):
  fig, ax = plt.subplots(figsize=plot_size)
  sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  # Add legend to explain the lines
  ax.legend()
  plt.tight_layout()
  plt.close(fig)
  return fig


fig1 = plot_feature_importances(importances, feat_names)
fig2 = plot_confusion_matrix(y_test, y_pred)


with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
  # Log the model
  mlflow.sklearn.log_model(sk_model=clf, input_example=X_test, artifact_path="model")
  mlflow.log_figure(fig1, "feature_importances.png")
  mlflow.log_figure(fig2, "confusion_matrix.png")
  # Log the metrics
  for label, metrics in classification_scores.items():
    if isinstance(metrics, dict):  # Only log precision/recall/F1/support
      for metric_name, value in metrics.items():
        metric_key = f"{label}_{metric_name}"
        mlflow.log_metric(metric_key, value)
        mlflow.log_param('n_estimators', n_estimators)
