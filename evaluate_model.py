import pandas as pd
import pickle
import os
import sys

model_path = 'model/model.pkl'
new_data_path = 'data/iris_new.csv'

if not os.path.exists(new_data_path):
    print("✅ New data not found. Skipping evaluation.")
    sys.exit(0)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

new_df = pd.read_csv(new_data_path)
X_new = new_df.drop('species', axis=1)
y_new = new_df['species']

accuracy = model.score(X_new, y_new)
print(f"Evaluation Accuracy on new data: {accuracy:.2f}")
