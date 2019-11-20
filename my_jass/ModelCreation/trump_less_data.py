from pathlib import Path

import pandas as pd
from tensorflow import keras
from tensorflow_core.python.keras.saving.save import load_model

# Import only a fraction of data for efficient testing
path_to_train_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\train\\output")
data = pd.read_csv("C:\\Users\\matth\\Documents\\DL4G\\jass-demo\\my_jass\\data\\2018_10_18_trump.csv", header=None)
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
data_columns_train = cards + forehand
data.drop('user', axis='columns', inplace=True)
print(data.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[data_columns_train], data.trump, test_size=0.25,
                                                    stratify=data.trump, random_state=42)
print(x_train)
print(y_train)

model = keras.Sequential()
model.add(keras.layers.Dense(37, activation='relu', input_shape=[37]))
model.add(keras.layers.Dense(37, activation='relu'))
model.add(keras.layers.Dense(37, activation='relu'))
model.add(keras.layers.Dense(37, activation='relu'))
model.add(keras.layers.Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
y_categorical = keras.utils.to_categorical(y_train)
y_categorical_test = keras.utils.to_categorical(y_test)
model.evaluate(x_test, y_categorical_test)

model.save(path_to_train_data / "deep_trump_model_v3.h5")

# model = load_model("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\train\\output\\deep_trump_model_v2.h5")
# print(model.evaluate(x_test, y_categorical_test))
