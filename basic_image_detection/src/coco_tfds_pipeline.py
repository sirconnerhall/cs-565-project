import os
os.environ["TFDS_DATA_DIR"] = "D:/tensorflow_datasets"

import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_example(example, image_size=(224, 224)):
    """
    example: {
      'image': tf.Tensor[H, W, 3],
      'image/filename': ...,
      'objects': {
          'bbox': [N, 4] (ymin, xmin, ymax, xmax normalized),
          'label': [N],
          'is_crowd': [N],
          ...
      },
      ...
    }
    """
    image = tf.image.resize(example["image"], image_size)
    image = tf.cast(image, tf.float32) / 255.0

    bboxes = example["objects"]["bbox"]
    labels = example["objects"]["label"]

    return image, {"bboxes": bboxes, "labels": labels}


def make_coco_dataset(split="train", batch_size=8, image_size=(224, 224)):
    ds, ds_info = tfds.load(
        "coco/2017",
        split=split,
        with_info=True,
        shuffle_files=True,
    )

    num_classes = ds_info.features["objects"]["label"].num_classes
    print("COCO num classes:", num_classes)

    autotune = tf.data.AUTOTUNE

    ds = (
        ds.map(lambda ex: preprocess_example(ex, image_size), num_parallel_calls=autotune)
          .shuffle(1024)
          .padded_batch(
              batch_size,
              padded_shapes=(
                  [*image_size, 3],        # image
                  {
                      "bboxes": [None, 4],  # variable number of boxes
                      "labels": [None],
                  },
              ),
              padding_values=(
                  0.0,                     # image padding value
                  {
                      "bboxes": 0.0,
                      "labels": tf.cast(0, tf.int64),
                  },
              ),
          )
          .prefetch(autotune)
    )

    return ds, ds_info

def make_coco_detection_dataset(split="train", batch_size=4, image_size=(512, 512)):
    """
    Returns a dataset tailored for object detection libraries like KerasCV.

    Each element is:
        {
          "images": [B, H, W, 3],
          "bounding_boxes": {
              "boxes":   [B, num_boxes, 4]  (rel_yxyx),
              "classes": [B, num_boxes],
          }
        }
    """
    ds, ds_info = tfds.load(
        "coco/2017",
        split=split,
        with_info=True,
        shuffle_files=True,
    )

    num_classes = ds_info.features["objects"]["label"].num_classes
    print("COCO num classes (detection):", num_classes)

    def _prep(example):
        image = tf.image.resize(example["image"], image_size)
        image = tf.cast(image, tf.float32) / 255.0

        boxes = example["objects"]["bbox"]      # [N,4] rel_yxyx
        classes = example["objects"]["label"]   # [N]

        return {
            "images": image,
            "bounding_boxes": {
                "boxes": boxes,
                "classes": tf.cast(classes, tf.float32),  # or int32; depends on lib
            },
        }

    autotune = tf.data.AUTOTUNE

    ds = (
        ds.map(_prep, num_parallel_calls=autotune)
          .ragged_batch(batch_size)   # Ragged is OK for KerasCV
          .prefetch(autotune)
    )

    return ds, ds_info


if __name__ == "__main__":
    train_ds, info = make_coco_dataset("train[:5%]", batch_size=4)
    for batch_images, batch_targets in train_ds.take(1):
        print("Images shape:", batch_images.shape)
        print("Bboxes shape:", batch_targets["bboxes"].shape)
        print("Labels shape:", batch_targets["labels"].shape)
        break
