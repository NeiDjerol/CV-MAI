import os
import random
import shutil

# =========================================
# НАСТРОЙКИ
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_RATIO = 0.85

# =========================================
# ПАПКИ TRAIN / VAL
# =========================================

TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_DIR, "train", "labels")

VAL_IMAGES = os.path.join(DATASET_DIR, "val", "images")
VAL_LABELS = os.path.join(DATASET_DIR, "val", "labels")

# Создаем папки

os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(TRAIN_LABELS, exist_ok=True)

os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

# =========================================
# СПИСОК ИЗОБРАЖЕНИЙ
# =========================================

images = [
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith(".jpg")
]

random.shuffle(images)

# =========================================
# SPLIT
# =========================================

train_count = int(len(images) * TRAIN_RATIO)

train_images = images[:train_count]
val_images = images[train_count:]

# =========================================
# ФУНКЦИЯ КОПИРОВАНИЯ
# =========================================

def copy_files(image_list, image_dest, label_dest):

    for image_file in image_list:

        image_src = os.path.join(IMAGES_DIR, image_file)

        label_file = os.path.splitext(image_file)[0] + ".txt"

        label_src = os.path.join(LABELS_DIR, label_file)

        # Копируем изображение
        shutil.copy(image_src, os.path.join(image_dest, image_file))

        # Копируем label
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(label_dest, label_file))
        else:
            print(f"Label не найден: {label_file}")

# =========================================
# КОПИРОВАНИЕ
# =========================================

copy_files(train_images, TRAIN_IMAGES, TRAIN_LABELS)

copy_files(val_images, VAL_IMAGES, VAL_LABELS)

# =========================================
# ГОТОВО
# =========================================

print("===================================")
print("DATASET SPLIT ГОТОВ")
print("===================================")

print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")

print()
print("Train path:")
print(TRAIN_IMAGES)

print()
print("Val path:")
print(VAL_IMAGES)