from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

path_to_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\test\\filtered\\csv")

# train data
data1 = pd.read_csv(path_to_data / '0001.csv', header=None)
data2 = pd.read_csv(path_to_data / '0002.csv', header=None)
data3 = pd.read_csv(path_to_data / '0003.csv', header=None)
data4 = pd.read_csv(path_to_data / '0004.csv', header=None)

data = pd.concat(
    [data1, data2, data3, data4],
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

print(data)

feature_columns = cards + forehand
x_train, x_test, y_train, y_test = train_test_split(data[feature_columns], data.trump, test_size=0.2,
                                                    stratify=data.trump, random_state=42)
print(x_train)
print(y_train)

model = keras.Sequential()
model.add(keras.layers.Dense(22, activation='relu', input_shape=[37]))
model.add(keras.layers.Dense(22, activation='relu'))
model.add(keras.layers.Dense(22, activation='relu'))
model.add(keras.layers.Dense(22, activation='relu'))
model.add(keras.layers.Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
y_categorical = keras.utils.to_categorical(y_train)
history = model.fit(x_train, y_categorical, validation_split=0.20, epochs=100, batch_size=100)

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

model.save("models/matt/filtered_deep_trump_model_v1.h5")
