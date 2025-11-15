import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras import layers

# IMPORTANT: this assumes coco_tfds_pipeline.py is in the same folder (src/)
from coco_tfds_pipeline import make_coco_dataset

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("////////////////Using GPU:", gpus)
else:
    print("////////////////No GPU detected — training will run on CPU.")

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def build_multilabel_model(
    image_size=(224, 224),
    num_classes=80,
    base_trainable=False,
    name="coco_multilabel_mobilenetv2",
):
    """
    Simple MobileNetV2 backbone + multi-label head.
    """
    inputs = keras.Input(shape=(*image_size, 3))

    base_model = keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    # Preprocess for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Multi-label: sigmoid instead of softmax
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=name)
    return model


def to_multilabel_batch(num_classes):
    """
    Convert batched COCO labels -> multi-hot vectors.

    Inputs:
        images: [batch, H, W, 3]
        targets["labels"]: [batch, max_boxes]

    Returns:
        images: [batch, H, W, 3]
        multi_hot: [batch, num_classes]
    """
    def _fn(images, targets):
        labels = targets["labels"]  # [batch, max_boxes]
        labels = tf.cast(labels, tf.int32)

        # Optional: if you modify your pipeline to pad with -1 instead of 0,
        # you can mask them out like:
        #   valid_mask = tf.not_equal(labels, -1)
        #   labels = tf.where(valid_mask, labels, tf.zeros_like(labels))

        one_hot = tf.one_hot(labels, num_classes)  # [batch, max_boxes, num_classes]
        multi_hot = tf.reduce_max(one_hot, axis=1)  # [batch, num_classes]

        return images, multi_hot

    return _fn


def main():
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"

    config = load_config(str(config_path))

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_split = config["train_split"]
    val_split = config["val_split"]
    learning_rate = config["learning_rate"]
    model_name = config["model_name"]
    models_dir = project_root / config["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # Make sure TFDS_DATA_DIR is pointing to D: (you already did this elsewhere)
    # os.environ["TFDS_DATA_DIR"] = "D:/tensorflow_datasets"

    print("Loading COCO via TFDS from:", os.environ.get("TFDS_DATA_DIR", "<default>"))

    # Use your existing pipeline to get ds + ds_info
    # Note: we pass batch_size here for convenience, but we'll re-batch below.
    train_ds_raw, train_info = make_coco_dataset(
        split=train_split,
        batch_size=batch_size,
        image_size=image_size,
    )
    val_ds_raw, _ = make_coco_dataset(
        split=val_split,
        batch_size=batch_size,
        image_size=image_size,
    )

    num_classes = train_info.features["objects"]["label"].num_classes
    print("COCO num classes:", num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    # Convert (images, {bboxes, labels}) → (images, multi_hot_labels)
    train_ds = (
        train_ds_raw
        .map(to_multilabel_batch(num_classes), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds_raw
        .map(to_multilabel_batch(num_classes), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # -----------------------------
    # Build and compile model
    # -----------------------------
    model = build_multilabel_model(
        image_size=image_size,
        num_classes=num_classes,
        base_trainable=False,
        name=model_name,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",   # multi-label
        metrics=[
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()

    # -----------------------------
    # Callbacks
    # -----------------------------
    best_model_path = models_dir / f"{model_name}_best.keras"
    last_model_path = models_dir / f"{model_name}_last.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=3,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=2,
            mode="max",
            verbose=1,
        ),
    ]

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save last model
    model.save(str(last_model_path))
    print(f"Saved last model to {last_model_path}")
    print(f"Best model (by val_auc) at {best_model_path}")


if __name__ == "__main__":
    main()
