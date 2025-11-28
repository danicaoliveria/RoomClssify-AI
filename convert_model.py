import pickle
import Orange
import os

input_path = os.path.join("model", "roomclassify.pkcls")
output_path = os.path.join("model", "model.pkl")

print("Loading Orange .pkcls:", input_path)

with open(input_path, "rb") as f:
    orange_model = pickle.load(f)

print("Loaded type:", type(orange_model))

# Extract the true sklearn model inside Orange model
try:
    sklearn_model = orange_model.skl_model     # Works for RandomForest, Tree, SVM, etc.
    print("Extracted scikit-learn model:", type(sklearn_model))
except:
    raise RuntimeError("❌ Could not extract sklearn model. This Orange model type is unsupported.")

# Save pure sklearn model
with open(output_path, "wb") as f:
    pickle.dump(sklearn_model, f)

print("✅ Conversion complete! Saved as:", output_path)
