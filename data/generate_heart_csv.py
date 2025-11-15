import pandas as pd
import numpy as np

np.random.seed(42)   # for reproducibility

rows = 200

data = {
    "age": np.random.randint(29, 77, rows),                   # typical heart dataset age range
    "sex": np.random.randint(0, 2, rows),                     # 0 = female, 1 = male
    "cp": np.random.randint(0, 4, rows),                      # chest pain type (0–3)
    "trestbps": np.random.randint(94, 200, rows),             # resting blood pressure
    "chol": np.random.randint(126, 564, rows),                # cholesterol mg/dl
    "thalach": np.random.randint(71, 202, rows),              # max heart rate
    "exang": np.random.randint(0, 2, rows),                   # exercise induced angina
}

# -------------------------
# Create a realistic target
# -------------------------
# Higher risk if:
# - age > 50
# - chol high
# - low thalach
# - chest pain cp==3
risk_score = (
    (data["age"] > 50).astype(int)
    + (data["chol"] > 240).astype(int)
    + (data["thalach"] < 150).astype(int)
    + (np.array(data["cp"]) == 3).astype(int)
)

# Convert risk score to binary target (0/1)
target = (risk_score >= 2).astype(int)

data["target"] = target

df = pd.DataFrame(data)
df.to_csv("data/heart.csv", index=False)

print("🔥 200-row heart.csv generated successfully in /data folder!")
