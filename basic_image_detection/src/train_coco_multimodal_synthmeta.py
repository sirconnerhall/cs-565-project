import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras import layers

from coco_tfds_pipeline import make_coco_dataset


# ----------------------------
# Config helpers
# ----------------------------

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


# ----------------------------
# Loss: focal with optional smoothing
# ----------------------------

def binary_focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.0):
    def _loss(y_true, y_pred):
        # Optional label smoothing (toward 0.5)
        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # BCE
        ce = -(y_true * tf.math.log(y_pred) +
               (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # alpha, modulating
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        loss = alpha_factor * modulating_factor * ce
        return tf.reduce_mean(loss)

    _loss.__name__ = "_loss"
    return _loss


# ----------------------------
# COCO labels -> multi-hot
# ----------------------------
from coco_multilabel_utils import to_multilabel_batch

# ----------------------------
# Synthetic metadata
# ----------------------------

def add_synthetic_metadata(num_locations=4):
    """
    Input:  (images, multi_hot_labels)
    Output: ((images, metadata_vec), multi_hot_labels)

    metadata = [brightness, is_day, is_night, one_hot_location(num_locations)]
            -> dim = 3 + num_locations
    """
    def _fn(images, multi_hot):
        # images: [B, H, W, 3], values in [0,1]
        batch_size = tf.shape(images)[0]

        # Brightness per image
        brightness = tf.reduce_mean(images, axis=[1, 2, 3])  # [B]
        brightness = tf.expand_dims(brightness, -1)          # [B,1]

        # Day/night flags based on brightness
        is_day = tf.cast(brightness[:, 0] > 0.5, tf.float32)     # [B]
        is_night = 1.0 - is_day

        is_day = tf.expand_dims(is_day, -1)       # [B,1]
        is_night = tf.expand_dims(is_night, -1)   # [B,1]

        # Fake location ID (0..num_locations-1) — random for now
        loc_ids = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=num_locations,
            dtype=tf.int32,
        )
        loc_onehot = tf.one_hot(loc_ids, num_locations)  # [B, num_locations]

        metadata = tf.concat(
            [brightness, is_day, is_night, loc_onehot],
            axis=1,
        )  # [B, 3 + num_locations]

        return (images, metadata), multi_hot

    return _fn


# ----------------------------
# Multimodal model
# ----------------------------

def build_multimodal_model(
    image_size=(224, 224),
    num_classes=80,
    metadata_dim=7,
    base_trainable=False,
    name="coco_multimodal_mobilenetv2",
):
    # Image branch
    image_input = keras.Input(shape=(*image_size, 3), name="image")
    base_model = keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    x_img = keras.applications.mobilenet_v2.preprocess_input(image_input)
    x_img = base_model(x_img, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    # Metadata branch
    meta_input = keras.Input(shape=(metadata_dim,), name="metadata")
    x_meta = layers.Dense(32, activation="relu")(meta_input)
    x_meta = layers.Dense(16, activation="relu")(x_meta)

    # Fuse
    x = layers.Concatenate()([x_img, x_meta])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[image_input, meta_input],
        outputs=outputs,
        name=name,
    )
    return model


def get_backbone(model):
    """
    Find the backbone sub-model inside the main model (first Keras Model layer).
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("Backbone not found in model.layers")


# ----------------------------
# Main training loop
# ----------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    config = load_config(str(config_path))

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_split = config["train_split"]
    val_split = config["val_split"]
    learning_rate = config["learning_rate"]
    model_name = config.get("model_name", "coco_multimodal_mobilenetv2")
    models_dir = project_root / config.get("models_dir", "models")

    # Focal / FT config with defaults
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.25)
    label_smoothing = config.get("label_smoothing", 0.0)
    fine_tune_backbone = config.get("fine_tune_backbone", True)
    fine_tune_after_epochs = config.get("fine_tune_after_epochs", 5)
    fine_tune_lr = config.get("fine_tune_lr", 1e-5)
    fine_tune_fraction = config.get("fine_tune_fraction", 0.8)

    # Synthetic metadata shape
    num_locations = 4
    metadata_dim = 3 + num_locations

    models_dir.mkdir(parents=True, exist_ok=True)

    print("Train split:", train_split)
    print("Val split:  ", val_split)
    print("Using focal loss (gamma, alpha):", focal_gamma, focal_alpha)
    print("Label smoothing:", label_smoothing)
    print("Fine-tune backbone:", fine_tune_backbone)

    # Datasets
    train_ds_raw, train_info = make_coco_dataset(
        split=train_split,
        batch_size=batch_size,
        image_size=image_size,
    )
    val_ds_raw, val_info = make_coco_dataset(
        split=val_split,
        batch_size=batch_size,
        image_size=image_size,
    )

    num_classes = train_info.features["objects"]["label"].num_classes
    print("COCO num classes:", num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    # (images, {bboxes, labels}) -> (images, multi_hot)
    train_ds = (
        train_ds_raw
        .map(to_multilabel_batch(num_classes), num_parallel_calls=AUTOTUNE)
        .map(add_synthetic_metadata(num_locations), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds_raw
        .map(to_multilabel_batch(num_classes), num_parallel_calls=AUTOTUNE)
        .map(add_synthetic_metadata(num_locations), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Model
    model = build_multimodal_model(
        image_size=image_size,
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        base_trainable=False,
        name=model_name,
    )

    loss_fn = binary_focal_loss(
        gamma=focal_gamma,
        alpha=focal_alpha,
        label_smoothing=label_smoothing,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()

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

    # -----------------------
    # Phase 1: head-only
    # -----------------------
    print("\n=== Phase 1: training with frozen backbone ===")
    first_phase_epochs = min(
        epochs, fine_tune_after_epochs if fine_tune_backbone else epochs
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=first_phase_epochs,
        callbacks=callbacks,
    )

    # -----------------------
    # Phase 2: fine-tune backbone
    # -----------------------
    if fine_tune_backbone and epochs > fine_tune_after_epochs:
        print("\n=== Phase 2: fine-tuning backbone ===")

        base_model = get_backbone(model)
        base_model.trainable = True

        num_layers = len(base_model.layers)
        freeze_until = int(num_layers * fine_tune_fraction)
        print(f"Backbone has {num_layers} layers, freezing first {freeze_until}.")

        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=loss_fn,
            metrics=[
                keras.metrics.BinaryAccuracy(name="bin_acc"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        remaining_epochs = epochs - fine_tune_after_epochs
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining_epochs,
            callbacks=callbacks,
        )

    model.save(str(last_model_path))
    print(f"\nSaved last model to {last_model_path}")
    print(f"Best model (by val_auc) at {best_model_path}")


if __name__ == "__main__":
    main()
