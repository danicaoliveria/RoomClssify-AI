import pickle
import Orange
import os

# Path to your Orange model
path = os.path.join("model", "roomclassify.pkcls")

print("Loading:", path)

# Load the model (Orange automatically loads sklearn estimators)
with open(path, "rb") as f:
    model = pickle.load(f)

print("Loaded type:", type(model))

# Save as pure scikit-learn model
output_path = os.path.join("model", "model.pkl")

with open(output_path, "wb") as f:
    pickle.dump(model, f)

print("Conversion complete! Saved:", output_path)
