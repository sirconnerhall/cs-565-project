# coco_multilabel_utils.py

import tensorflow as tf


# ----------------------------
# Binary focal loss (multi-label)
# ----------------------------

def binary_focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.0):
    """
    Binary focal loss for multi-label classification.

    gamma: focusing parameter, typically 2.0
    alpha: weight for positive examples, typically 0.25
    label_smoothing: e.g. 0.1 to soften targets slightly

    This matches the loss used in training and evaluation scripts so it
    can also be used via custom_objects when loading a saved model.
    """
    def _loss(y_true, y_pred):
        # Optional label smoothing toward 0.5
        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # Standard BCE
        cross_entropy = -(y_true * tf.math.log(y_pred) +
                          (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # p_t is probability of the true class
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # alpha balancing factor
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        # modulating factor
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        loss = alpha_factor * modulating_factor * cross_entropy
        return tf.reduce_mean(loss)

    # Important for loading with custom_objects
    _loss.__name__ = "_loss"
    return _loss


# ----------------------------
# COCO -> multi-hot image labels
# ----------------------------

def to_multilabel_batch(num_classes):
    """
    Convert batched COCO labels -> multi-hot vectors, ignoring padded boxes.

    Inputs from make_coco_dataset:
        images: [B, H, W, 3]
        targets["bboxes"]: [B, max_boxes, 4]  (ymin, xmin, ymax, xmax) in [0,1]
        targets["labels"]: [B, max_boxes]

    Padded boxes from coco_tfds_pipeline.py use zero coords, so we treat
    any box with zero area as invalid and ignore its label.

    Returns:
        images:   [B, H, W, 3]
        multi_hot:[B, num_classes]
    """
    def _fn(images, targets):
        labels = tf.cast(targets["labels"], tf.int32)      # [B, max_boxes]
        bboxes = targets["bboxes"]                         # [B, max_boxes, 4]

        # Compute valid (non-padded) boxes: area > 0
        ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=-1)
        valid = (ymax > ymin) & (xmax > xmin)              # [B, max_boxes]

        # Clip labels to valid class range (padding labels don't matter once masked)
        labels_clipped = tf.clip_by_value(labels, 0, num_classes - 1)

        one_hot = tf.one_hot(labels_clipped, num_classes)  # [B, max_boxes, C]

        # Zero out one-hot vectors for invalid boxes
        mask = tf.cast(valid, tf.float32)[..., tf.newaxis] # [B, max_boxes, 1]
        one_hot = one_hot * mask

        # Reduce over boxes: any box of class j -> that entry is 1
        multi_hot = tf.reduce_max(one_hot, axis=1)         # [B, C]

        return images, multi_hot

    return _fn


# ----------------------------
# HELP ME
# ----------------------------

def run_extended_sanity_checks(
    train_ds_raw,
    train_ds,
    num_classes,
    image_size,
    grid_size,
    model,
):
    print("\n================ EXTENDED SANITY CHECK ================")
    print(f"image_size: {image_size}, grid_size: {grid_size}, num_classes: {num_classes}")

    # -----------------------------
    # 1) RAW DATASET CHECK
    # -----------------------------
    print("\n[1] Checking raw dataset (images + bboxes + labels)")

    got_raw_batch = False
    for images, targets in train_ds_raw.take(1):
        got_raw_batch = True

        print("  raw images shape:   ", images.shape)
        print("  raw bboxes shape:   ", targets["bboxes"].shape)
        print("  raw labels shape:   ", targets["labels"].shape)

        # Basic image stats
        img_min = tf.reduce_min(images)
        img_max = tf.reduce_max(images)
        print("  image min/max:      ", float(img_min.numpy()), float(img_max.numpy()))

        # Check first sample's boxes
        b0 = targets["bboxes"][0]
        l0 = targets["labels"][0]

        print("  first sample bboxes shape:", b0.shape)
        if tf.size(b0) > 0:
            print("    bbox ymin/xmin/ymax/xmax min:",
                  float(tf.reduce_min(b0).numpy()),
                  "max:",
                  float(tf.reduce_max(b0).numpy()))
        else:
            print("    no bboxes in first sample")

        print("  first sample labels shape:", l0.shape)
        if tf.size(l0) > 0:
            print("    labels min/max:",
                  int(tf.reduce_min(l0).numpy()),
                  int(tf.reduce_max(l0).numpy()))
        else:
            print("    no labels in first sample")

        break

    if not got_raw_batch:
        raise RuntimeError("[SANITY] train_ds_raw is EMPTY — no batches produced.")

    # -----------------------------
    # 2) ENCODED GRID DATASET CHECK
    # -----------------------------
    print("\n[2] Checking encoded grid dataset")

    got_enc_batch = False
    for images_enc, grid_true in train_ds.take(1):
        got_enc_batch = True

        print("  enc images shape:   ", images_enc.shape)
        print("  grid_true shape:    ", grid_true.shape)

        # Expected depth
        expected_depth = 5 + num_classes
        actual_depth = grid_true.shape[-1]
        print("  expected depth:", expected_depth, "actual depth:", actual_depth)

        if actual_depth is None:
            print("  [WARN] grid_true last dim is None (dynamic), "
                  "but should be 5 + num_classes.")
        elif actual_depth != expected_depth:
            raise RuntimeError(
                f"[SANITY] grid_true last dimension mismatch: "
                f"expected {expected_depth}, got {actual_depth}"
            )

        # Objectness stats
        obj_true = grid_true[..., 0]
        obj_min = float(tf.reduce_min(obj_true).numpy())
        obj_max = float(tf.reduce_max(obj_true).numpy())
        obj_counts = tf.reduce_sum(tf.cast(obj_true > 0.5, tf.float32), axis=[1, 2])  # per-sample count

        print("  objectness min/max:", obj_min, obj_max)
        print("  object cells per sample (first batch):", obj_counts.numpy())

        # Forward pass + loss
        preds = model(images_enc, training=False)
        print("  preds shape:        ", preds.shape)

        # Check for NaNs
        if tf.math.reduce_any(tf.math.is_nan(preds)):
            raise RuntimeError("[SANITY] NaNs detected in model predictions.")
        if tf.math.reduce_any(tf.math.is_nan(grid_true)):
            raise RuntimeError("[SANITY] NaNs detected in grid_true targets.")

        break

    if not got_enc_batch:
        raise RuntimeError("[SANITY] encoded train_ds is EMPTY — no batches produced.")

    print("\n================ SANITY CHECK PASSED =================\n")
