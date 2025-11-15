import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def load_and_preprocess_image(img_path, image_size):
    """
    Loads a single image and preprocesses it exactly like training:
    - Resize
    - Convert to float32
    - Normalize to [0, 1]
    - Add batch dimension
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(image_size)
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)   # [1, H, W, 3]

def load_class_names():
    label_file = r"D:\tensorflow_datasets\coco\2017\1.1.0\objects-label.labels.txt"
    with open(label_file, "r") as f:
        return [line.strip() for line in f.readlines()]



def main():
    # -----------------------------
    # Locate config and model
    # -----------------------------
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    config = load_config(str(config_path))

    image_size = tuple(config["image_size"])
    model_name = config["model_name"]
    models_dir = project_root / config["models_dir"]

    best_model_path = models_dir / f"{model_name}_best.keras"
    last_model_path = models_dir / f"{model_name}_last.keras"

    if best_model_path.exists():
        model_path = best_model_path
    elif last_model_path.exists():
        model_path = last_model_path
        print(f"[WARNING] Best model not found. Using last model instead: {model_path}")
    else:
        raise FileNotFoundError("Could not find best or last model.")

    print("Loading model:", model_path)
    model = keras.models.load_model(model_path)

    # -----------------------------
    # COCO label names -- too lazy to link to the db
    # -----------------------------
    class_names = load_class_names()


    # -----------------------------
    # Ask user for image path
    # -----------------------------
    img_path = input("Enter path to image: ").strip()
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    print("Loading image:", img_path)
    image_batch = load_and_preprocess_image(img_path, image_size)

    # -----------------------------
    # Predict
    # -----------------------------
    preds = model.predict(image_batch)[0]   # shape: [num_classes]
    threshold = 0.5

    pred_indices = np.where(preds >= threshold)[0]
    pred_labels = [(class_names[i], float(preds[i])) for i in pred_indices]

    print("\n--- Prediction Results ---")
    if not pred_labels:
        print("No classes predicted above threshold", threshold)
    else:
        for label, score in pred_labels:
            print(f"{label:20s}  score={score:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
