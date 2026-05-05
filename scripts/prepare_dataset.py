#!/usr/bin/env python3
import numpy as np

def split_dataset(train_images, seed=42, val_ratio=0.15):
    rng = np.random.RandomState(seed)
    rng.shuffle(train_images)

    split = int(len(train_images) * (1 - val_ratio))
    return train_images[:split], train_images[split:]

def main():
    raw_path = ...
    out_path = ...

    train_imgs = load_train_images(raw_path)

    train_split, val_split = split_dataset(train_imgs)

    process_to_tfrecord(train_split, "train.record")
    process_to_tfrecord(val_split, "val.record")

    process_test_set(raw_val_set)

    create_representative_dataset(train_split)
