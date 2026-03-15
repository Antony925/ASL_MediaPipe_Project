# -*- coding: utf-8 -*-

#test_model.py
#Скрипт виконує:
#1) Завантаження моделі
#2) Робота з камерою
#3) Додавання класифікованої літери в масив
#4) Вивід класифікованої літери на екран та/або у активне вікно (браузер, месенджер, тощо)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import time
import pyautogui


# Завантаження CNN и LabelEncoder
model = tf.keras.models.load_model("models/asl_model.h5")
le = joblib.load("models/label_encoder.pkl")


# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
current_phrase = ""
last_prediction = ""
last_added_time = 0
min_hold_time = 0.75
max_width = 800

def draw_multiline_text(frame, text, start_x, start_y, font, scale, color, thickness, line_height=30):
    words = text.split(" ")
    line = ""
    y = start_y
    for word in words:
        test_line = line + ("" if line == "" else " ") + word
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w > max_width:
            cv2.putText(frame, line, (start_x, y), font, scale, color, thickness)
            line = word
            y += line_height
        else:
            line = test_line
    if line:
        cv2.putText(frame, line, (start_x, y), font, scale, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    prediction = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z

        features = []
        for lm in hand_landmarks.landmark:
            x = lm.x - base_x
            y = lm.y - base_y
            z = lm.z - base_z
            features.append([x, y, z])
        features = np.array(features).reshape(1,21,3)
        features = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True)+1e-6)

        probs = model.predict(features)
        class_idx = np.argmax(probs)
        prediction = le.inverse_transform([class_idx])[0]

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"Letter: {prediction}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    # Додавання літери та друк у активне вікно
    current_time = time.time()
    if prediction:
        if prediction != last_prediction:
            last_added_time = current_time
            last_prediction = prediction
        elif current_time - last_added_time >= min_hold_time:
            if prediction == "space":
                current_phrase += " "
                pyautogui.write(" ")
            elif prediction == "del":
                current_phrase = current_phrase[:-1]
                pyautogui.press("backspace")
            else:
                current_phrase += prediction
                pyautogui.write(prediction)
            last_added_time = current_time

    draw_multiline_text(frame, current_phrase, start_x=50, start_y=100,
                        font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(0,255,0),
                        thickness=2, line_height=40)

    cv2.imshow("ASL CNN Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('c'):
        current_phrase = ""
        last_prediction = ""
        last_added_time = time.time()

cap.release()
cv2.destroyAllWindows()