import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from manual_feature_prediction import get_accuracy, get_confusion, get_total_confusion, apply_single_threshold

def build_simple_cnn(input_shape):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_total_cnn(input_shape):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(4, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == '__main__':

    batch_size = 32
    epochs = 10
    validation_split = 0.1

    data = pd.read_csv("features_and_labels.csv")

    files = list(data["filename"])
    split_point = int(len(files) * .8)

    images = []
    for file in files:
        # Open in grayscale
        image = cv2.imread("good_crops/" + file, 0)
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 255
        images.append(image)

    images = np.expand_dims(images, -1)

    circle_model = build_simple_cnn(images[0].shape)
    hands_model = build_simple_cnn(images[0].shape)
    digits_model = build_simple_cnn(images[0].shape)
    total_model = build_total_cnn(images[0].shape)

    train_images = images[:split_point]
    test_images = images[split_point:]

    circle_labels = list(data["ClockContour"])
    hands_labels = list(data["ClockHands"])
    digits_labels = list(data["ClockNumbers"])
    total_labels = list(data["ClockTotal"])

    circle_train_labels = circle_labels[:split_point]
    circle_test_labels = circle_labels[split_point:]
    hands_train_labels = hands_labels[:split_point]
    hands_test_labels = hands_labels[split_point:]
    digits_train_labels = digits_labels[:split_point]
    digits_test_labels = digits_labels[split_point:]
    total_train_labels = total_labels[:split_point]
    total_test_labels = total_labels[split_point:]

    circle_model.fit(train_images, np.array(circle_train_labels), batch_size=batch_size, epochs=epochs)
    print("Circle model trained!")
    hands_model.fit(train_images, np.array(hands_train_labels), batch_size=batch_size, epochs=epochs)
    print("Hands model trained!")
    digits_model.fit(train_images, np.array(digits_train_labels), batch_size=batch_size, epochs=epochs)
    print("Digits model trained!")

    total_model.fit(train_images, to_categorical(total_train_labels, 4), batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    print("Total model trained!")

    circle_predictions = np.squeeze(circle_model.predict(test_images), axis=1)
    hands_predictions = np.squeeze(hands_model.predict(test_images), axis=1)
    digits_predictions = np.squeeze(digits_model.predict(test_images), axis=1)
    total_predictions = np.argmax(total_model.predict(test_images), axis=1)
    print(circle_predictions)

    circularity_threshold = 0

    accuracies = []
    tprs = []
    fprs = []
    tnrs = []
    fnrs = []
    f1s = []
    threshes = []

    dummy_thresh = 0
    best_circle_score = 0
    for i in range(100):
        threshed_predictions = apply_single_threshold(circle_predictions, dummy_thresh)

        accuracies.append(get_accuracy(circle_test_labels, threshed_predictions))
        f1 = f1_score(circle_test_labels, threshed_predictions, pos_label=0)
        f1s.append(f1)
        tpr, fpr, tnr, fnr = get_confusion(circle_test_labels, threshed_predictions)
        tprs.append(tpr)
        fprs.append(fpr)
        tnrs.append(tnr)
        fnrs.append(fnr)
        threshes.append(dummy_thresh)

        if f1 > best_circle_score:
            best_circle_score = f1
            circularity_threshold = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(circle_predictions, circularity_threshold)

    fig = plt.figure(0, (15, 15))
    fig.suptitle("Performance using simple CNN", fontsize=24)

    plt.subplot(3, 3, 1)
    plt.plot(threshes, accuracies, label="Accuracy", color="red")
    plt.plot(threshes, tnrs, label="True Negative Rate", color="orange")
    plt.plot(threshes, fnrs, label="False Negative Rate", color="brown")
    plt.vlines(circularity_threshold, 0, 1, colors="gray")
    plt.legend()
    plt.xlabel("Threshold Value")
    plt.title("Circularity Accuracy, TNR, and FNR")

    plt.subplot(3, 3, 2)
    plt.plot(threshes, f1s, label="F1 Score", color="green")
    plt.plot(threshes, tprs, label="True Positive Rate", color="blue")
    plt.plot(threshes, fprs, label="False Positive Rate", color="purple")
    plt.vlines(circularity_threshold, 0, 1, colors="gray")
    plt.xlabel("Threshold Value")
    plt.title("Circularity Harmonic Mean, TPR, and FPR")
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(fprs, tprs, label="ROC Curve", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Circularity ROC Curve")
    plt.legend()

    # -------------------------------------------------------------------------------------------------------------------
    # Phase 2 - determine values for hands ratio

    lower_hands_threshold = 0

    accuracies = []
    tprs = []
    fprs = []
    tnrs = []
    fnrs = []
    f1s = []
    threshes = []

    dummy_thresh = 0
    best_hands_score = 0
    for i in range(100):
        threshed_predictions = apply_single_threshold(hands_predictions, dummy_thresh)

        accuracies.append(get_accuracy(hands_test_labels, threshed_predictions))
        f1 = f1_score(hands_test_labels, threshed_predictions, pos_label=0)
        f1s.append(f1)
        tpr, fpr, tnr, fnr = get_confusion(hands_test_labels, threshed_predictions)
        tprs.append(tpr)
        fprs.append(fpr)
        tnrs.append(tnr)
        fnrs.append(fnr)
        threshes.append(dummy_thresh)

        if f1 > best_hands_score:
            best_hands_score = f1
            lower_hands_threshold = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(hands_predictions, lower_hands_threshold)

    plt.subplot(3, 3, 4)
    plt.plot(threshes, accuracies, label="Accuracy", color="red")
    plt.plot(threshes, tnrs, label="True Negative Rate", color="orange")
    plt.plot(threshes, fnrs, label="False Negative Rate", color="brown")
    plt.vlines(lower_hands_threshold, 0, 1, colors="gray")
    plt.xlabel("Threshold Value")
    plt.title("Hands Accuracy, TNR, and FNR")
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(threshes, f1s, label="F1 Score", color="green")
    plt.plot(threshes, tprs, label="True Positive Rate", color="blue")
    plt.plot(threshes, fprs, label="False Positive Rate", color="purple")
    plt.vlines(lower_hands_threshold, 0, 1, colors="gray")
    plt.xlabel("Threshold Value")
    plt.title("Hands Harmonic Mean, TPR, and FPR")
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(fprs, tprs, label="ROC Curve", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Hands ROC Curve")
    plt.legend()

    # Phase 3 - determine values for digits

    digits_threshold = 0

    accuracies = []
    tprs = []
    fprs = []
    tnrs = []
    fnrs = []
    f1s = []
    threshes = []

    dummy_thresh = 0
    best_digits_score = 0
    for i in range(100):
        threshed_predictions = apply_single_threshold(digits_predictions, dummy_thresh)

        accuracies.append(get_accuracy(digits_test_labels, threshed_predictions))
        f1 = f1_score(digits_test_labels, threshed_predictions, pos_label=0)
        f1s.append(f1)
        tpr, fpr, tnr, fnr = get_confusion(digits_test_labels, threshed_predictions)
        tprs.append(tpr)
        fprs.append(fpr)
        tnrs.append(tnr)
        fnrs.append(fnr)
        threshes.append(dummy_thresh)

        if f1 > best_digits_score:
            best_digits_score = f1
            digits_threshold = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(digits_predictions, digits_threshold)

    plt.subplot(3, 3, 7)
    plt.plot(threshes, accuracies, label="Accuracy", color="red")
    plt.plot(threshes, tnrs, label="True Negative Rate", color="orange")
    plt.plot(threshes, fnrs, label="False Negative Rate", color="brown")
    plt.vlines(digits_threshold, 0, 1, colors="gray")
    plt.legend()
    plt.xlabel("Threshold Value")
    plt.title("Digits Accuracy, TNR, and FNR")

    plt.subplot(3, 3, 8)
    plt.plot(threshes, f1s, label="F1 Score", color="green")
    plt.plot(threshes, tprs, label="True Positive Rate", color="blue")
    plt.plot(threshes, fprs, label="False Positive Rate", color="purple")
    plt.vlines(digits_threshold, 0, 1, colors="gray")
    plt.xlabel("Threshold Value")
    plt.title("Digits Harmonic Mean, TPR, and FPR")
    plt.legend()

    plt.subplot(3, 3, 9)
    plt.plot(fprs, tprs, label="ROC Curve", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Digits ROC Curve")
    plt.legend()

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig("simple_cnn_performance.jpg")
    plt.show()

    combined_predictions = np.array(apply_single_threshold(circle_predictions, circularity_threshold)) \
                           + np.array(apply_single_threshold(hands_predictions, lower_hands_threshold)) \
                           + np.array(apply_single_threshold(digits_predictions, digits_threshold))

    print(combined_predictions)
    total_accuracy = get_accuracy(total_test_labels, total_predictions)
    type1, type2 = get_total_confusion(total_test_labels, total_predictions)
    print("Total accuracy for single model: ", total_accuracy)
    print("Type 1 misclassification rate: ", type1)
    print("Type 2 misclassification rate: ", type2)
    print("")

    combined_accuracy = get_accuracy(total_test_labels, combined_predictions)
    type1, type2 = get_total_confusion(total_test_labels, combined_predictions)
    print("Total accuracy for combined little models: ", combined_accuracy)
    print("Type 1 misclassification rate: ", type1)
    print("Type 2 misclassification rate: ", type2)
    print("")

    print("Contour F-Score: ", best_circle_score)
    print("Hands F-Score: ", best_hands_score)
    print("Digits F-Score: ", best_digits_score)