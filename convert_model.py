import pickle
import os

# Path to your Orange .pkcls
input_path = os.path.join("model", "roomclassify.pkcls")
output_path = os.path.join("model", "model.pkl")

print("Loading Orange .pkcls:", input_path)

try:
    with open(input_path, "rb") as f:
        model = pickle.load(f)  # Orange stores sklearn models directly

    print("Loaded model type:", type(model))

    # Save as standard scikit-learn pickle
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Conversion complete! Saved as:", output_path)

except Exception as e:
    print("❌ Error converting model:", e)
