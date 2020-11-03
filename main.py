# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os

import tensorflow as tf
import tensorboard
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import keras
from sklearn.metrics import accuracy_score

# %%
model_name = "alexnet"
print("Model used: {}".format(model_name))
data=[]
labels=[]

if model_name == "lenet5":
    height = 28
    width = 28
elif model_name == "alexnet":
    height = 227
    width = 227
else:
    height = 30
    width = 30

n_channels    = 3
n_classes     = 43
n_inputs = height * width * n_channels

# Import images to appropriate data structures
for i in range(n_classes) :
    path = "Train/{}/".format(i)
    images = os.listdir(path)
    for a in images:
        try:
            img = Image.open(path + a)
            img = img.resize((width, height))   # Ensure all images are sized equally
            img = np.array(img)
            data.append(img)
            labels.append(i)
        except:
            print("Error loading image")

data = np.array(data)
labels = np.array(labels)


# Randomizes and splits data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, n_classes)
y_val  = to_categorical(y_val , n_classes)

# %%
# Define CNN model
def tut_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(n_classes, activation='softmax'))

    # Compilation of the model
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    return model

def lenet5():

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 3)))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    # Compilation of the model
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    return model

def alexnet():
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(n_classes, activation='softmax')
    ])


    # Compilation of the model
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    return model


if model_name == "lenet5":
    model = lenet5()
elif model_name == "alexnet":
    model = alexnet()
else:
    model = tut_model()


# %%
# Train and validate model
epochs = 20
history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data=(X_val, y_val))

# %%
# Save model
model.save("Models/{}/model_{}_epochs".format(model_name,epochs))

# %%
# Grab the epoch list and the training+validation loss+accuracy lists.
epoch = history.epoch
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# Plot accuracy and loss
plt.figure(0)
plt.plot(accuracy, label='training accuracy')
plt.plot(val_accuracy, label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Models/{}/accuracy_{}_epochs.png'.format(model_name,epochs))

plt.figure(1)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('Models/{}/loss_{}_epochs.png'.format(model_name,epochs))