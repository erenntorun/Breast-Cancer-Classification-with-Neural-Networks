#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:57:28 2024

@author: eren
"""


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


veri = pd.read_csv("/mnt/c/USERS/CASPER/Downloads/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data")
# '2' ise iyi huylu '4' ise kötü huylu demek.

veri.replace('?', -99999, inplace=True)
#veri.drop(['id'], axis=1)
veriyeni = veri.drop(['1000025'], axis=1)

imp = SimpleImputer(missing_values=-99999, strategy="mean")
veriyeni = imp.fit_transform(veriyeni)


giris = veriyeni[:,0:8]
cikis = veriyeni[:,9]

print(giris.shape)  # 8 Tane özellik verilmiş bunlara göre 9. yu bulucaz.
print(cikis.shape)

model = Sequential()
model.add(Dense(256, input_dim=8))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(giris, cikis, epochs=50, batch_size=32, validation_split=0.13)

# Örnek tahmin verisi
tahmin = np.array([10,5,5,3,6,7,7,10]).reshape(1, 8)

# Tahmini yap
prediction = model.predict(tahmin)

# En yüksek olasılığa sahip sınıfı seçme
class_prediction = np.argmax(prediction, axis=1)

# 0 ve 1 değerlerini 2 ve 4 olarak dönüştürme
if class_prediction[0] == 0:
    class_prediction[0] = 2  # 0'ı 2'ye dönüştür
else:
    class_prediction[0] = 4  # 1'i 4'e dönüştür

print(class_prediction)  # Sonuç 2 veya 4 olacak


