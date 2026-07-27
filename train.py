"""
train.py
--------
Trains the Smart Sorting vegetable freshness classifier using transfer
learning on top of MobileNetV2.

============================================================
CONCEPTS EXPLAINED (read this before running the script)
============================================================

1. CNNs (Convolutional Neural Networks)
   A CNN learns to recognise visual patterns — edges, textures, shapes — by
   sliding small filters (kernels) across an image. Early layers learn
   simple patterns (edges, colours); deeper layers combine these into
   complex concepts (a wilted leaf, a brown spot, a vegetable's shape).

2. Transfer Learning
   Training a CNN from scratch needs millions of images and days of GPU
   time. Instead, we reuse a network (MobileNetV2) that Google already
   trained on ImageNet (1.4M images, 1000 classes). Its early/middle layers
   already know how to detect general visual features (edges, textures,
   colour gradients) — features that are just as useful for spotting a
   rotten spot on a tomato as they are for recognising a cat. We only need
   to train a new "head" (final layers) to map those features to OUR
   20 classes.

3. Freezing Layers
   During the first training phase, we "freeze" (lock) MobileNetV2's
   pretrained weights so they don't change. We only train the new
   classification head we added on top. This is fast and prevents
   destroying the useful pretrained features with a small dataset.

4. Fine-Tuning
   After the head has learned reasonably well, we "unfreeze" the last
   several layers of MobileNetV2 and continue training with a VERY low
   learning rate. This lets the model slightly adjust its high-level
   features to be more specific to vegetables, squeezing out extra accuracy.

5. Data Augmentation
   We artificially create variations of training images (rotate, flip,
   zoom, shift brightness) on-the-fly. This exposes the model to more
   visual variety than the raw dataset alone provides, reducing overfitting.

6. Callbacks
   - EarlyStopping: stops training automatically once validation loss stops
     improving, preventing wasted epochs and overfitting.
   - ReduceLROnPlateau: shrinks the learning rate when validation loss
     plateaus, helping the model converge more precisely.
   - ModelCheckpoint: saves only the BEST model (based on validation
     accuracy) seen during training, not just whatever the last epoch was.

7. Metrics
   - Accuracy: % of correct predictions overall. Can be misleading if
     classes are imbalanced.
   - Precision: of everything predicted "Rotten", how many actually were?
   - Recall: of everything that actually was "Rotten", how many did we catch?
   - F1 Score: harmonic mean of precision & recall — a balanced single number.
   - Confusion Matrix: a grid showing exactly which classes get confused
     with which — invaluable for debugging a multi-class model.
   - ROC Curve / AUC: shows the trade-off between true-positive rate and
     false-positive rate at different thresholds (computed per-class here
     since this is a multi-class problem).

============================================================
"""

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuration — tweak these based on your dataset size / GPU
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 15          # phase 1: train only the new head
EPOCHS_FINE_TUNE = 10     # phase 2: fine-tune top layers of MobileNetV2
FINE_TUNE_AT = 100        # unfreeze layers from this index onward
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "smart_sorting_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(MODEL_DIR, "roc_curve.png")

os.makedirs(MODEL_DIR, exist_ok=True)


def check_gpu():
    """Log whether a GPU is available — training on CPU works but is much slower."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU(s) detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning(
            "No GPU detected. Training will run on CPU and will be slow. "
            "If running locally, consider using Google Colab (free GPU) instead."
        )


def build_data_generators():
    """
    Build train/val/test data generators.

    Training generator uses augmentation (rotation, flips, zoom, shifts,
    brightness) to expose the model to more visual variety.
    Validation/test generators only rescale — we NEVER augment val/test data,
    since we want those to reflect real, unmodified images.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_model(num_classes: int) -> Model:
    """
    Build the transfer-learning model:
      MobileNetV2 (frozen, pretrained on ImageNet)
        -> GlobalAveragePooling2D  (flattens feature maps to a vector)
        -> BatchNormalization      (stabilises training)
        -> Dense(256, relu)        (learns task-specific combinations)
        -> Dropout(0.4)            (reduces overfitting)
        -> Dense(num_classes, softmax)  (final classification layer)
    """
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,       # exclude ImageNet's original 1000-class head
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def get_callbacks(phase: str):
    return [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        TensorBoard(log_dir=os.path.join("tensorboard_logs", phase)),
    ]


def plot_training_history(history_head, history_fine=None):
    """Plot accuracy & loss curves across training (and fine-tuning, if run)."""
    acc = history_head.history["accuracy"]
    val_acc = history_head.history["val_accuracy"]
    loss = history_head.history["loss"]
    val_loss = history_head.history["val_loss"]

    if history_fine:
        acc += history_fine.history["accuracy"]
        val_acc += history_fine.history["val_accuracy"]
        loss += history_fine.history["loss"]
        val_loss += history_fine.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close()
    logger.info(f"Saved training curves to {HISTORY_PLOT_PATH}")


def evaluate_model(model, test_gen, class_labels):
    """Run final evaluation on the held-out test set: classification report,
    confusion matrix, and per-class ROC curves."""
    logger.info("Running evaluation on test set...")
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_labels, digits=4
    )
    logger.info(f"Classification Report:\n{report}")
    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_labels, yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    logger.info(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")

    # ROC curves (one-vs-rest, since this is multi-class)
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_labels))))
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()
    logger.info(f"Saved ROC curves to {ROC_CURVE_PATH}")


def main():
    logger.info("=== Smart Sorting: Model Training Started ===")
    check_gpu()

    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not os.path.isdir(d) or not os.listdir(d):
            raise FileNotFoundError(
                f"Expected populated dataset directory at '{d}'. "
                f"See docs/02_dataset.md for how to download and organise the dataset."
            )

    train_gen, val_gen, test_gen = build_data_generators()
    num_classes = train_gen.num_classes
    class_labels = list(train_gen.class_indices.keys())

    # Persist class index mapping so predict.py / app.py stay in sync with
    # however Keras happened to order the folders on this machine.
    index_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(index_to_class, f, indent=2)
    logger.info(f"Saved class indices ({num_classes} classes) to {CLASS_INDICES_PATH}")

    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary(print_fn=logger.info)

    # ---------------- Phase 1: train the new head, base frozen ----------------
    logger.info("=== Phase 1: Training classification head (base frozen) ===")
    start = time.time()
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks("phase1_head"),
    )
    logger.info(f"Phase 1 completed in {time.time() - start:.1f}s")

    # ---------------- Phase 2: fine-tune top layers of MobileNetV2 ----------------
    logger.info("=== Phase 2: Fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False  # keep earlier (more generic) layers frozen

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    start = time.time()
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=get_callbacks("phase2_finetune"),
    )
    logger.info(f"Phase 2 completed in {time.time() - start:.1f}s")

    # Save final model explicitly too (ModelCheckpoint already saved the best one)
    model.save(MODEL_PATH)
    logger.info(f"Final model saved to {MODEL_PATH}")

    plot_training_history(history_head, history_fine)
    evaluate_model(model, test_gen, class_labels)

    logger.info("=== Training pipeline complete ===")


if __name__ == "__main__":
    main()
