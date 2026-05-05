#!/usr/bin/env python3

from pathlib import Path
import random
import tensorflow as tf
from PIL import Image

class DataSetup():

    def __init__(self, data_dir="/kaggle/input/datasets/manhhoangvan/cropsweed-5680103/CropsWeed_5680103"):
        
        DATA_DIR = Path(data_dir)

        images = list((DATA_DIR / "images").glob("*.jpeg"))

        print("Total images:", len(images))


    def check_health(self):
        bad_files = 0
        bad_lines_total = 0
        files_with_missing_labels = 0

        for img_path in images:
            label_path = DATA_DIR / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                files_with_missing_labels += 1
                break

            for line in label_path.read_text().splitlines():
                if len(line.strip().split()) != 5:
                    bad_files += 1
                    bad_lines_total += 1
                    break

        print(f"Files with bad labels: {bad_files}")
        print(f"Total bad lines: {bad_lines_total}")
        print(f"Total with missing labels: {files_with_missing_labels}")


    def prepare_split(self):
        random.shuffle(images)

        split = int(0.8 * len(images))
        train_imgs = images[:split]
        val_imgs = images[split:]

        print("Train:", len(train_imgs), "Val:", len(val_imgs))


    def write_label_map(self):
        label_map = {
            0: "crop",
            1: "weed"
        }

        label_map_path = Path("/kaggle/working/label_map.pbtxt")

        with label_map_path.open("w") as f:
            for k, v in label_map.items():
                f.write(f"item {{\n id: {k+1}\n name: '{v}'\n}}\n")

        print("label_map.pbtxt created")





def create_tf_example(image_path: Path, label_path: Path):
    encoded_jpg = image_path.read_bytes()

    image = Image.open(image_path)
    width, height = image.size

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, x, y, w, h = map(float, parts)

        xmin = (x - w/2) * width
        xmax = (x + w/2) * width
        ymin = (y - h/2) * height
        ymax = (y + h/2) * height

        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)

        classes.append(int(cls) + 1)
        classes_text.append(label_map[int(cls)].encode())

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_path).encode()])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_path).encode()])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

'done'







def write_tfrecord(output_path: Path, image_list):
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for img_path in image_list:
            label_path = DATA_DIR / "labels" / (img_path.stem + ".txt")

            if not label_path.exists():
                continue

            tf_example = create_tf_example(img_path, label_path)
            writer.write(tf_example.SerializeToString())

train_record = Path("/kaggle/working/train.record")
val_record = Path("/kaggle/working/val.record")

write_tfrecord(train_record, train_imgs)
write_tfrecord(val_record, val_imgs)

print("TFRecords created")



raw_dataset = tf.data.TFRecordDataset(str(train_record))

for raw_record in raw_dataset.take(1):
    print("TFRecord sample loaded successfully")



sample_img = images[0]
sample_label = DATA_DIR / "labels" / (sample_img.stem + ".txt")

print(sample_img)
print(sample_label)
print(sample_label.exists())
print(sample_label.read_text()[:100])
