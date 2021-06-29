from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.svm import SVR, SVC
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from manual_feature_prediction import get_accuracy, get_confusion, apply_single_threshold, get_total_confusion


# # densely sample keypoints
def sample_kp(shape, stride, size):
    kp = []
    for i in range(0, shape[0], stride[0]):
        for j in range(0, shape[1], stride[0]):
            kp.append(cv2.KeyPoint(i, j, size))

    return np.array(kp)


#
# # extract vocabulary of SIFT features
def extract_vocabulary(raw_data, key_point, n):
    sift = cv2.SIFT_create()
    descriptors = np.array([])
    kp, des = sift.compute(raw_data[0], key_point)
    descriptors = des
    for img in raw_data[1:]:
        kp, des = sift.compute(img, key_point)
        descriptors = np.append(descriptors, des, axis=0)

    vocabulary = KMeans(n_clusters=n).fit(descriptors)
    return vocabulary

#
# # extract Bag of SIFT Representation of images
def extract_feat(raw_data, vocabulary, key_point, n):
    feat = []
    sift = cv2.SIFT_create()
    for img in raw_data:
        # Create a blank feature vector the size of the number of clusters
        features = np.zeros((n), dtype=np.float32)
        kp, des = sift.compute(img, key_point)
        labels = vocabulary.predict(des)
        for label in labels:
            features[label] += 1
        # Normalize the features
        features = features / len(labels)
        feat.append(features)

    return np.array(feat)


def get_SIFT_descriptors(image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des

def get_SIFT_batch(images):
    keypoints = []
    descriptors = []
    for image in images:
        kp, des = get_SIFT_descriptors(image)
        keypoints.append(kp)
        descriptors.append(des)

    return keypoints, descriptors

if __name__ == '__main__':

    circle_model = SVR(kernel="rbf")
    hands_model = SVR(kernel="rbf")
    digits_model = SVR(kernel="rbf")
    total_model = SVC(kernel="rbf")

    data = pd.read_csv("features_and_labels.csv")

    files = list(data["filename"])
    split_point = int(len(files) * .8)

    images = []
    for file in files:
        # Open in grayscale
        image = cv2.imread("good_crops/" + file, 0)
        image = cv2.resize(image, (256, 256))
        images.append(image)

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

    #train_kps, train_des = get_SIFT_batch(train_images)
    #test_kps, test_des = get_SIFT_batch(test_images)

    skp = sample_kp((train_images[0].shape[0], train_images[0].shape[1]), (16,16), 8)
    vocabulary = extract_vocabulary(train_images, skp, 100)
    train_des = extract_feat(train_images, vocabulary, skp, 100)
    test_des = extract_feat(test_images, vocabulary, skp, 100)

    circle_model.fit(train_des, circle_train_labels)
    print("Circle model trained!")
    hands_model.fit(train_des, hands_train_labels)
    print("Hands model trained!")
    digits_model.fit(train_des, digits_train_labels)
    print("Digits model trained!")

    total_model.fit(train_des, total_train_labels)
    print("Total model trained!")

    circle_predictions = circle_model.predict(test_des)
    hands_predictions = hands_model.predict(test_des)
    digits_predictions = digits_model.predict(test_des)
    total_predictions = total_model.predict(test_des)

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
    fig.suptitle("Performance using SIFT Features and SVM", fontsize=24)

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

    plt.savefig("SIFT-SVM_feature_performance.jpg")
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