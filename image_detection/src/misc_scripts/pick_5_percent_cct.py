import json
import random
from pathlib import Path

ANNOT = Path(r"d:\datasets\cct\annotations\caltech_images_20210113.json")
OUT = Path(r"D:\datasets\cct\annotations\sampled_5pct_all.txt")

PERCENT = 0.05  # 5 percent

data = json.load(open(ANNOT, "r"))

images = data["images"]  # standard COCO-style list
num_total = len(images)
num_sample = int(num_total * PERCENT)

print("Total images:", num_total)
print("Sampling:", num_sample)

sampled = random.sample(images, num_sample)

with open(OUT, "w") as f:
    for img in sampled:
        f.write(img["file_name"] + "\n")

print("Wrote sampled file list to:", OUT)
