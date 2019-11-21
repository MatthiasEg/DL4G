import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.saving.save import load_model


path_to_train_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\train\\output")
path_to_test_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\test\\trump_csv")

# train data
data1 = pd.read_csv(path_to_train_data / 'rnd_01.csv', header=None)
data2 = pd.read_csv(path_to_train_data / 'rnd_02.csv', header=None)
data3 = pd.read_csv(path_to_train_data / 'rnd_03.csv', header=None)
data4 = pd.read_csv(path_to_train_data / 'rnd_04.csv', header=None)
data5 = pd.read_csv(path_to_train_data / 'rnd_05.csv', header=None)
data6 = pd.read_csv(path_to_train_data / 'rnd_06.csv', header=None)
data7 = pd.read_csv(path_to_train_data / 'rnd_07.csv', header=None)
data8 = pd.read_csv(path_to_train_data / 'rnd_08.csv', header=None)
data9 = pd.read_csv(path_to_train_data / 'rnd_09.csv', header=None)
data10 = pd.read_csv(path_to_train_data / 'rnd_10.csv', header=None)
data11 = pd.read_csv(path_to_train_data / 'rnd_11.csv', header=None)

# test data
data12 = pd.read_csv(path_to_test_data / 'rnd_01.csv', header=None)
data13 = pd.read_csv(path_to_test_data / 'rnd_02.csv', header=None)
data14 = pd.read_csv(path_to_test_data / 'rnd_03.csv', header=None)
data15 = pd.read_csv(path_to_test_data / 'rnd_04.csv', header=None)

# val data
data16 = pd.read_csv(path_to_test_data / 'rnd_01.csv', header=None)
data17 = pd.read_csv(path_to_test_data / 'rnd_02.csv', header=None)
data18 = pd.read_csv(path_to_test_data / 'rnd_03.csv', header=None)
data19 = pd.read_csv(path_to_test_data / 'rnd_04.csv', header=None)

data = pd.concat(
    [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,
     data16, data17, data18, data19],
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

data.columns = cards + forehand + user + trump

# remove user information
data.drop('user', axis='columns', inplace=True)

feature_columns = cards + forehand
x_train, x_test, y_train, y_test = train_test_split(data[feature_columns], data.trump, test_size=0.2,
                                                    stratify=data.trump, random_state=42)
print(x_train)
print(y_train)


model = load_model("./models/matt/deep_trump_model_v4.h5")



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
y_categorical = keras.utils.to_categorical(y_train)
history = model.fit(x_train, y_categorical, validation_split=0.20, epochs=100, batch_size=10000)

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
