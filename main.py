import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from tabulate import tabulate

"""
Predictive Observability with Machine Learning

This project uses machine learning (Isolation Forest) to detect anomalies in simulated log data. 
It analyzes features like severity level, message length, and time of day, flagging even subtle issues hidden in DEBUG or INFO logs.
"""

# --- Load and parse logs ---
log_file_path = "simulated_logs.txt"  # Path to log file

pattern = r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+) (.+)$"
data = []

with open(log_file_path, "r") as file:
    for line in file:
        match = re.match(pattern, line.strip())
        if match:
            timestamp = f"{match.group(1)} {match.group(2)}"
            level = match.group(3)
            message = match.group(4)
            data.append([timestamp, level, message])

# --- Create DataFrame ---
df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Feature engineering ---
# Convert severity levels to numerical scores
level_mapping = {
    "TRACE": 1,
    "DEBUG": 2,
    "INFO": 3,
    "WARN": 4,
    "ERROR": 5,
    "FATAL": 6,
}
df["level_score"] = df["level"].map(level_mapping)
df["message_length"] = df["message"].apply(len)

# Add time-based features
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

# --- Anomaly detection ---
features = ["level_score", "message_length", "hour", "weekday"]
model = IsolationForest(contamination=0.1, random_state=42)
df["anomaly"] = model.fit_predict(df[features])
df["is_anomaly"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

"""
What does the contamination parameter mean? 
It specifies the proportion of outliers in the data set. A contamination of 0.1 means we expect 10% of the data to be outliers.

Why is 0.1 a good choice? 
It is a common starting point for many datasets, but it can be adjusted based on domain knowledge or prior analysis.

What does the random_state parameter do?
It ensures reproducibility of results by controlling the randomness in the algorithm.

Why is it set to 42? 
42 is often used as a "default" seed in examples and tutorials, but any integer can be used.
"""

# --- Filter and sort anomalies ---
anomalies = df[df["is_anomaly"] == "Anomaly"].sort_values("timestamp")

# --- Output ---
# Cleaned-up display of anomalies
print("\n Detected Anomalies:\n")
print(tabulate(
    anomalies[["timestamp", "level", "message", "level_score", "message_length"]],
    headers=["Timestamp", "Level", "Message", "Level Score", "Msg Length"],
    tablefmt="fancy_grid",
    showindex=False
))
