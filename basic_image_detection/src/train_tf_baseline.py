import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def main():
    # ----------------------------
    # Basic config
    # ----------------------------
    data_dir = pathlib.Path("data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    image_size = (224, 224)
    batch_size = 32
    seed = 42
    epochs = 10

    # Make sure output dir exists
    os.makedirs("models", exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("Using data from:", data_dir.resolve())

    # ----------------------------
    # Load datasets from folders
    # ----------------------------
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .cache()
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # ----------------------------
    # Data augmentation pipeline
    # ----------------------------
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    # ----------------------------
    # Build transfer learning model
    # ----------------------------
    # Use MobileNetV2 as a lightweight pretrained backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False  # freeze for a simple baseline

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    inputs = keras.Input(shape=(*image_size, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="tf_baseline_mobilenetv2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ----------------------------
    # Training
    # ----------------------------
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/baseline_best.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save final model
    model.save("models/baseline_final.keras")
    print("Saved models to ./models")

    # ----------------------------
    # Quick evaluation
    # ----------------------------
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

    # Optionally: print simple per-class counts
    print("\nClass index mapping:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
