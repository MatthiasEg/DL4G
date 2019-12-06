import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.saving.save import load_model


path_to_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\test\\filtered\\csv")

# train data
data1 = pd.read_csv(path_to_data / '0001.csv', header=None)
data2 = pd.read_csv(path_to_data / '0002.csv', header=None)
data3 = pd.read_csv(path_to_data / '0003.csv', header=None)

data = pd.concat(
    [data1, data2, data3],
    axis=0, ignore_index=True)

# data = data.head(1000)

# print(data.shape)

cards = [
    # Diamonds
    'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
    # Hearts
    'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
    # Spades
    'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
    # Clubs
    'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
]

# Forehand (yes = 1, no = 0)
forehand = ['FH']

user = ['user']
trump = ['trump']

data.columns = cards + forehand + trump
feature_columns = cards + forehand

x_train, x_test, y_train, y_test = train_test_split(data[feature_columns], data.trump, test_size=0.2,
                                                    stratify=data.trump, random_state=42)
print(x_train)
print(y_train)


model = load_model("./models/matt/filtered_deep_trump_model_v1.h5")



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
y_categorical = keras.utils.to_categorical(y_train)
history = model.fit(x_train, y_categorical, validation_split=0.20, epochs=2000, batch_size=10000)

y_categorical_test = keras.utils.to_categorical(y_test)
print(model.evaluate(x_test, y_categorical_test))

plt.plot(history.history['loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['Train', 'Val'], loc='upper left')

plt.show()

# model.save(path_to_train_data / "deep_trump_model_v5.h5")
model.save("models/matt/deep_trump_model_v4_refitted_adam.h5")
