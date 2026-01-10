"""MobileNetV2 transfer-learning stack for an Indian food classifier.

The module keeps everything needed for an end-to-end experiment in one place:
* Builds tf.data pipelines with gentle augmentation so CPU-only laptops can cope.
* Freezes a MobileNetV2 backbone, adds a lean head, and trains with Adam.
* Tracks train/val/test splits, evaluates on the hold-out set, and saves artifacts.
* Ships a tiny inference helper so you can grab top-k predictions from any image.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "indian_food"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

IMAGE_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
SEED = 1337
MODEL_OUTPUT_PATH = MODEL_DIR / "indian_food_mobilenet.keras"
CLASS_NAMES_PATH = MODEL_DIR / "indian_food_class_names.json"


def _assert_dirs_exist() -> None:
    """Guard clause: bail out early if the expected split folders are missing."""
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected directory '{split_dir}' was not found. Create the"
                " folder and populate it with class-specific sub-directories."
            )


def _load_dataset(directory: Path, shuffle: bool) -> tf.data.Dataset:
    """Wrap ``image_dataset_from_directory`` with the defaults we care about."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def build_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Create cached + prefetched datasets for the three splits and return class names."""
    _assert_dirs_exist()

    train_ds = _load_dataset(TRAIN_DIR, shuffle=True)
    val_ds = _load_dataset(VAL_DIR, shuffle=False)
    test_ds = _load_dataset(TEST_DIR, shuffle=False)

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    def configure(ds: tf.data.Dataset, shuffle_buffer: int | None = None) -> tf.data.Dataset:
        if shuffle_buffer:
            ds = ds.shuffle(shuffle_buffer)
        return ds.cache().prefetch(buffer_size=autotune)

    train_ds = configure(train_ds, shuffle_buffer=1000)
    val_ds = configure(val_ds)
    test_ds = configure(test_ds)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """Wire up the MobileNetV2 backbone with our lightweight classification head."""
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Phase 1: only teach the new head.

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="indian_food_mobilenetv2")
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train() -> None:
    """Run training, checkpointing, evaluation, and artifact export in one go."""
    train_ds, val_ds, test_ds, class_names = build_datasets()
    num_classes = len(class_names)
    model = build_model(num_classes)

    checkpoint_cb = callbacks.ModelCheckpoint(
        str(MODEL_OUTPUT_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    print("Training complete. Evaluating on the held-out test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f} | Test loss: {test_loss:.3f}")

    # Also persist the final weights so an artifact exists even if val never improves.
    model.save(MODEL_OUTPUT_PATH, overwrite=True)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"Saved model to {MODEL_OUTPUT_PATH}")
    print(f"Saved class-name index to {CLASS_NAMES_PATH}")

    # Ship the raw history so downstream notebooks can plot loss/accuracy curves.
    history_path = MODEL_DIR / "training_history.json"
    history_path.write_text(json.dumps(history.history, indent=2))
    print(f"Training history written to {history_path}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_class_names(path: Path = CLASS_NAMES_PATH) -> list[str]:
    """Read the class-index file that the training loop wrote."""
    return json.loads(path.read_text())


def _prepare_image(image_path: Path) -> np.ndarray:
    """Resize + preprocess a single RGB image so MobileNetV2 can consume it."""
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict_top_k(
    model_path: Path,
    image_path: Path,
    class_names: Iterable[str],
    k: int = 3,
    model: tf.keras.Model | None = None,
) -> list[tuple[str, float]]:
    """Return the k most likely labels, paired with their softmax probabilities."""
    model = model or tf.keras.models.load_model(model_path)
    processed = _prepare_image(image_path)
    probs = model.predict(processed, verbose=0)[0]

    class_names = list(class_names)
    k = min(k, len(class_names))
    top_values, top_indices = tf.math.top_k(probs, k=k)
    return [
        (class_names[idx], float(score))
        for idx, score in zip(top_indices.numpy(), top_values.numpy())
    ]


if __name__ == "__main__":
    train()
