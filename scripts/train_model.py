# -*- coding: utf-8 -*-

#train_model.py
#Скрипт виконує:
#1) Завантаження координат ключових точок
#2) Нормалізацію даних
#3) Кодування міток класів
#4) Побудову MLP моделі
#5) Навчання нейронної мережі
#6) Збереження моделі


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import joblib
import os


# Завантаження датасету координат
data = pd.read_csv("data/keypoints.csv", header=None)
X = data.iloc[:,1:].values.reshape(-1,21,3)
y = data.iloc[:,0].values


# Нормалізація координат кожної руки
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True) + 1e-6
X = (X - X_mean) / X_std


# LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")


# Розділ на тренувальну і тестову виборку
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Визначення MLP моделі
model = Sequential([
    Flatten(input_shape=(21,3)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Навчання
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/asl_model.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)


# Оцінка на тестовій виборці
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")