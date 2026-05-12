import cv2
import numpy as np
import os

# =========================================
# ПУТИ
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "video.mp4")

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# НАСТРОЙКИ
# =========================================

FRAME_SKIP = 15

# =========================================
# АУГМЕНТАЦИИ
# =========================================

def brighten(image, value=35):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def darken(image, value=35):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = np.clip(v.astype(np.int16) - value, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def add_noise(image):

    noise = np.random.normal(0, 15, image.shape).astype(np.int16)

    noisy = image.astype(np.int16) + noise

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def blur_image(image):

    return cv2.GaussianBlur(image, (5, 5), 0)


# =========================================
# ЧТЕНИЕ ВИДЕО
# =========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Ошибка: видео не открылось!")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Количество кадров в видео: {frame_count}")

frame_id = 0
saved_count = 0

# =========================================
# ОБРАБОТКА
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Берем только каждый N-й кадр
    if frame_id % FRAME_SKIP != 0:
        frame_id += 1
        continue

    # =====================================
    # АУГМЕНТАЦИИ
    # =====================================

    augmented = {

        "original": frame,

        "flip_horizontal": cv2.flip(frame, 1),

        "bright": brighten(frame),

        "dark": darken(frame),

        "blur": blur_image(frame),

        "noise": add_noise(frame)
    }

    # =====================================
    # СОХРАНЕНИЕ
    # =====================================

    for aug_name, aug_image in augmented.items():

        filename = f"frame_{frame_id}_{aug_name}.jpg"

        save_path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(save_path, aug_image)

        saved_count += 1

    print(f"Кадр {frame_id} обработан")

    frame_id += 1

# =========================================
# ЗАВЕРШЕНИЕ
# =========================================

cap.release()

print()
print("=====================================")
print("ГОТОВО")
print(f"Сохранено изображений: {saved_count}")
print(f"Папка: {OUTPUT_DIR}")
print("=====================================")