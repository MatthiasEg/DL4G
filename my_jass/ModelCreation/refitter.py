from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.keras import regularizers
import tensorflow as tf

## define paths to data files
from tensorflow_core.python.keras.saving import load_model

path_to_train_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\train\\filtered\\card\\csv")
path_to_test_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\test\\filtered\\card\\csv")
path_to_val_data = Path("C:\\Users\\matth\\Documents\\DL4G\\jass-data\\split\\val\\filtered\\card\\csv")

# train data
data_train1 = pd.read_csv(path_to_train_data / '0001.csv', header=None)
data_train2 = pd.read_csv(path_to_train_data / '0002.csv', header=None)
data_train3 = pd.read_csv(path_to_train_data / '0003.csv', header=None)
data_train4 = pd.read_csv(path_to_train_data / '0004.csv', header=None)
data_val1 = pd.read_csv(path_to_val_data / '0001.csv', header=None)
data_val2 = pd.read_csv(path_to_val_data / '0002.csv', header=None)

data_train = pd.concat([data_train1, data_train2, data_train3, data_train4, data_val1, data_val2], axis=0)

# test data
data_test1 = pd.read_csv(path_to_test_data / '0001.csv', header=None)
data_test2 = pd.read_csv(path_to_train_data / '0002.csv', header=None)

data_test = pd.concat([data_test1, data_test2])

# colums used for extracting x and y values. The same effect could be achieved with train_test_split-Method, but
# since we already have different files, we dont need to split the files using this method.
# data_X_columns = cards + forehand
# data_Y_colums = trump

x_train = data_train[data_train.columns[0:82]]
y_train = data_train[data_train.columns[82]]
print(x_train.head())
print(y_train.head())

x_test = data_test[data_test.columns[0:82]]
y_test = data_test[data_test.columns[82]]

# x_val = data_val[data_val.columns[0:82]]
# y_val = data_val[data_val.columns[82]]


model = load_model("./models/matt/card/best_card_model_68.h5")

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

y_train_categorical = to_categorical(y_train)
history = model.fit(x_train, y_train_categorical, epochs=120, batch_size=100)


y_categorical_test = keras.utils.to_categorical(y_test)
print(model.evaluate(x_test, y_categorical_test))

y_test_categorical = to_categorical(y_test)
print (model.evaluate(x_test, y_test_categorical))

# plt.plot(history.history['loss'])
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend(['Train', 'Val'], loc='upper left')

plt.plot(history.history['loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['Train'], loc='upper left')

plt.show()

model.save("./models/matt/card/model.h5")
