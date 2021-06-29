from os import path
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_model(num_classes):
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ]
    )
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model

def train_model(model, train_data, test_data, train_labels, test_labels):
    model.fit(train_data, train_labels, batch_size=128, epochs=30, validation_split=0.1)

    score = model.evaluate(test_data, test_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

##################################################################################################################

def area_of_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return 0
    else:
        return w * h

##################################################################################################################

def preprocess_data(data):
    processed = []
    for vis in data:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(vis)
        boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
        small_boxes = []
        for box in boxes:
            if box[2] > 80 or box[3] > 80:
                continue
            else:
                small_boxes.append(box)

        for box in small_boxes[:]:
            for other_box in small_boxes[:]:
                if other_box == box:
                    continue
                aoi = area_of_intersection(box, other_box)
                if aoi != 0:
                    box_area = box[2] * box[3]
                    other_box_area = other_box[2] * other_box[3]
                    try:
                        if box_area >= other_box_area:
                            small_boxes.remove(other_box)
                        else:
                            small_boxes.remove(box)
                    except:
                        pass

        small_boxes = list(set(small_boxes))
        number_crops = []
        for box in small_boxes:
            crop = vis[box[1]:box[1] + box[3] + 1, box[0]:box[0] + box[2] + 1]

            side_length = max(box[2], box[3]) + 6
            background = np.full((side_length, side_length), 255.0)

            x1 = int((side_length - box[2]) / 2)
            y1 = int((side_length - box[3]) / 2)
            x2 = x1 + box[2] + 1
            y2 = y1 + box[3] + 1

            background[y1:y2, x1:x2] = crop
            processed.append(cv2.resize(background, (28, 28)))

    return np.array(processed)

#############################################################################################

# the data, split between train and test sets
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = preprocess_data(x_train)
print("preproceesed training images")
x_test = preprocess_data(x_test)
print("preprocessed testing images")

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = build_model(10)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

train_model(model, x_train, x_test, y_train, y_test)
model.save("augmented_mnist_digit_classifier.h5")