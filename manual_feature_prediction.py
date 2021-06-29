import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

def get_accuracy(labels, predictions):
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct += 1

    return correct / len(labels)

#-----------------------------------------------------------------------------------------------------------------------


def get_confusion(labels, predictions):
    fps = 0
    fns = 0
    tps = 0
    tns = 0

    for i in range(len(labels)):
        if labels[i] == 0 and predictions[i] == 0:
            tps += 1
        if labels[i] == 0 and predictions[i] == 1:
            fns += 1
        if labels[i] == 1 and predictions[i] == 1:
            tns += 1
        if labels[i] == 1 and predictions[i] == 0:
            fps += 1

    if tps + fns == 0:
        tpr = 0
        fnr = 0
    else:
        tpr = tps/(tps + fns)
        fnr = fns/(fns + tps)

    if fps + tns == 0:
        fpr = 0
        tnr = 0
    else:
        fpr = fps/(fps + tns)
        tnr = tns/(tns + fps)

    return tpr, fpr, tnr, fnr

#-----------------------------------------------------------------------------------------------------------------------


def get_total_confusion(labels, predictions):
    # type 1 is false positive, type 2 is false negative
    type1 = 0
    type2 = 0
    for i in range(len(labels)):
        if predictions[i] < labels[i]:
            type1 += 1
        if labels[i] < predictions[i]:
            type2 += 1

    return type1/len(labels), type2/len(labels)

#-----------------------------------------------------------------------------------------------------------------------


def apply_single_threshold(predictions, threshold):
    threshed = []
    for prediction in predictions:
        if prediction >= threshold:
            threshed.append(1)
        else:
            threshed.append(0)

    return threshed

#-----------------------------------------------------------------------------------------------------------------------


def apply_double_threshold(predictions, lower, upper):
    threshed = []
    for prediction in predictions:
        if lower <= prediction and prediction < upper:
            threshed.append(1)
        else:
            threshed.append(0)

    return threshed

#-----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    data = pd.read_csv("features_and_labels.csv")

    # Phase 1 - determine values for circularity

    circularity_threshold = 0
    circle_labels = list(data["ClockContour"])
    print(sum(circle_labels), " of ", len(circle_labels), " circle labels are positive")
    circle_predictions = list(data["Circularity"])

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

        accuracies.append(get_accuracy(circle_labels, threshed_predictions))
        f1 = f1_score(circle_labels, threshed_predictions, pos_label=0)
        f1s.append(f1)
        tpr, fpr, tnr, fnr = get_confusion(circle_labels, threshed_predictions)
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
    data["CirclePredictions"] = threshed_predictions

    fig = plt.figure(0, (15, 15))
    fig.suptitle("Performance using Hand Extracted Features", fontsize=24)

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

    #-------------------------------------------------------------------------------------------------------------------
    # Phase 2 - determine values for hands ratio

    lower_hands_threshold = 0
    upper_hands_threshold = 0.8
    hands_labels = list(data["ClockHands"])
    print(sum(hands_labels), " of ", len(hands_labels), " hands labels are positive")
    hands_predictions = list(data["HandsRatio"])

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

        accuracies.append(get_accuracy(hands_labels, threshed_predictions))
        f1 = f1_score(hands_labels, threshed_predictions, pos_label=0)
        f1s.append(f1)
        tpr, fpr, tnr, fnr = get_confusion(hands_labels, threshed_predictions)
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
    data["HandsPredictions"] = threshed_predictions

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
    digits_labels = list(data["ClockNumbers"])
    print(sum(digits_labels), " of ", len(digits_labels), " digits labels are positive")
    digits_predictions = list(data["Digits"])

    accuracies = []
    tprs = []
    fprs = []
    tnrs = []
    fnrs = []
    f1s = []
    threshes = []

    dummy_thresh = 0
    best_digits_score = 0
    for i in range(int(max(digits_predictions))):
        threshed_predictions = apply_single_threshold(digits_predictions, dummy_thresh)

        accuracies.append(get_accuracy(digits_labels, threshed_predictions))
        f1 = f1_score(digits_labels, threshed_predictions, pos_label=0)
        f1s.append(f1)
        tpr, fpr, tnr, fnr = get_confusion(digits_labels, threshed_predictions)
        tprs.append(tpr)
        fprs.append(fpr)
        tnrs.append(tnr)
        fnrs.append(fnr)
        threshes.append(dummy_thresh)

        if f1 > best_digits_score:
            best_digits_score = f1
            digits_threshold = dummy_thresh

        dummy_thresh += 1

    threshed_predictions = apply_single_threshold(digits_predictions, digits_threshold)
    data["DigitPredictions"] = threshed_predictions

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

    plt.savefig("manual_feature_performance.jpg")
    plt.show()

    combined_predictions = np.array(apply_single_threshold(circle_predictions, circularity_threshold)) \
                           + np.array(apply_single_threshold(hands_predictions, lower_hands_threshold)) \
                           + np.array(apply_single_threshold(digits_predictions, digits_threshold))


    data["PredictionTotal"] = combined_predictions

    total_labels = list(data["ClockTotal"])
    #print(combined_predictions)
    #total_accuracy = get_accuracy(total_labels, total_predictions)
    #type1, type2 = get_total_confusion(total_labels, total_predictions)
    #print("Total accuracy for single model: ", total_accuracy)
    #print("Type 1 misclassification rate: ", type1)
    #print("Type 2 misclassification rate: ", type2)
    #print("")

    combined_accuracy = get_accuracy(total_labels, combined_predictions)
    type1, type2 = get_total_confusion(total_labels, combined_predictions)
    print("Total accuracy for combined little models: ", combined_accuracy)
    print("Type 1 misclassification rate: ", type1)
    print("Type 2 misclassification rate: ", type2)
    print("")

    print("Contour F-Score: ", best_circle_score)
    print("Hands F-Score: ", best_hands_score)
    print("Digits F-Score: ", best_digits_score)

    data.to_csv("features_and_labels.csv")

