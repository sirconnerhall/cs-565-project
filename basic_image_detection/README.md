# AI Image Detection – COCO (TF / TFDS)

This project is a stepping stone toward a full image detection + metadata pipeline.
Right now it:

- Downloads and caches MS-COCO 2017 via TensorFlow Datasets on `D:/tensorflow_datasets`.
- Uses a TF/Keras model to do **multi-label classification** on COCO images
  (which classes are present in the image?).
- Provides scripts to visualize bounding boxes and verify the pipeline.

## Setup

```bash
pip install -r requirements.txt
