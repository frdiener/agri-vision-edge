import tensorflow as tf
import numpy as np
from .phenobench_loader import PhenoBench
from .preprocessing import process_sample

def pil_to_numpy(img):
    return np.array(img, dtype=np.uint8)


def create_tf_example(image, boxes, labels):
    height, width = image.shape[:2]

    encoded = tf.io.encode_jpeg(image).numpy()

    xmins = [b[0] for b in boxes]
    ymins = [b[1] for b in boxes]
    xmaxs = [b[2] for b in boxes]
    ymaxs = [b[3] for b in boxes]

    classes = labels
    classes_text = [
        b"crop" if l == 1 else b"weed"
        for l in labels
    ]

    feature = {
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),

        "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),

        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def build_record(target, dataset: PhenoBench, with_tqdm: bool = False):

    writer = tf.io.TFRecordWriter(target)

    written = 0
    skipped = 0

    # optional tqdm
    if with_tqdm:
        from tqdm import tqdm
        iterator = tqdm(range(len(dataset)))
    else:
        iterator = range(len(dataset))

    for i in iterator:
        sample = dataset[i]

        image = pil_to_numpy(sample["image"])
        instances = sample["plant_instances"]
        semantics = sample["semantics"]

        image_resized, boxes, labels = process_sample(
            image=image,
            instances=instances,
            semantics=semantics,
            size=320,
            allowed_classes=(1, 2),
            min_area=20,
        )

        # skip empty images (important!)
        if len(boxes) == 0:
            skipped += 1
            continue

        example = create_tf_example(image_resized, boxes, labels)
        writer.write(example.SerializeToString())

        written += 1

    writer.close()

    print(f"written: {written}, skipped: {skipped}")
