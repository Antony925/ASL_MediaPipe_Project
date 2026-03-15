# -*- coding: utf-8 -*-

#extract_keypoints.py
#Скрипт виконує:
#1) Завантаження зображень датасету ASL
#2) Виявлення руки за допомогою MediaPipe Hands
#3) Витяг координат 21 ключової точки
#4) Нормалізацію координат відносно зап'ястка
#5) Збереження результатів у CSV файл

import os
import cv2
import csv
import mediapipe as mp

DATASET_PATH = r"C:\Users\Antony\Desktop\Dyplom\ASL_Alphabet_Dataset\asl_alphabet_train\asl_alphabet_train"
OUTPUT_CSV = "data/keypoints.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

data = []
labels = os.listdir(DATASET_PATH)

for label in labels:
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue
    print("Processing label:", label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            if len(landmarks.landmark) != 21:
                continue
            base_x = landmarks.landmark[0].x
            base_y = landmarks.landmark[0].y
            base_z = landmarks.landmark[0].z
            row = [label]
            for lm in landmarks.landmark:
                x = lm.x - base_x
                y = lm.y - base_y
                z = lm.z - base_z
                row.extend([x, y, z])
            data.append(row)

os.makedirs("data", exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Dataset saved: {OUTPUT_CSV}, total samples: {len(data)}")