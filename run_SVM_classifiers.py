from sklearn.svm import SVR, SVC
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from itertools import chain, combinations
from manual_feature_prediction import apply_single_threshold, get_accuracy, get_confusion

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
########################################################################################################################

def scale_to_binary(values):
    scaled_values = []
    min_val = np.min(values)
    max_val = np.max(values)
    for value in values:
        scaled = (value - min_val) / (max_val - min_val)
        scaled_values.append(scaled)

    return scaled_values

########################################################################################################################

def normalize(df):
    df["CenterDeviation"] = df["CenterDeviation"] / np.max(df["CenterDeviation"])
    df["HandsAngle"] = df["HandsAngle"] / np.max(df["HandsAngle"])
    df["IntersectDistance"] = df["IntersectDistance"] / np.max(df["IntersectDistance"])
    df["NumComponents"] = df["NumComponents"] / np.max(df["NumComponents"])
    df["DigitAngleMean"] = df["DigitAngleMean"] / np.max(df["DigitAngleMean"])
    df["DigitAngleStd"] = df["DigitAngleStd"] / np.max(df["DigitAngleStd"])
    df["DigitAreaMean"] = df["DigitAreaMean"] / np.max(df["DigitAreaMean"])
    df["DigitAreaStd"] = df["DigitAreaStd"] / np.max(df["DigitAreaStd"])
    df["ExtraDigits"] = df["ExtraDigits"] / np.max(df["ExtraDigits"])
    df["MissingDigits"] = df["MissingDigits"] / np.max(df["MissingDigits"])

    return df

########################################################################################################################

def run_classifiers(df, columns, label_name):

    labels = df[[label_name]].to_numpy()
    split_point = int(len(labels) * .8)

    train_labels = labels[:split_point]
    test_labels = labels[split_point:]

    rows_list = []
    combos = list(powerset(columns))[1:]
    # Test all feature combos
    for feature_set in combos:
        model = SVR(kernel="rbf")

        data = df[list(feature_set)].to_numpy()
        train_data = data[:split_point]
        test_data = data[split_point:]

        model.fit(train_data, train_labels.ravel())

        predictions = model.predict(test_data)

        # Record predictions for single features
        if feature_set == ('Circularity',):
            df["Pred_Circularity"] = model.predict(data)
        if feature_set == ('RemovedPoints',):
            df["Pred_RemovedPoints"] = model.predict(data)
        if feature_set == ('CenterDeviation',):
            df["Pred_CenterDeviation"] = model.predict(data)

        if feature_set == ('HandsAngle',):
            df["Pred_HandsAngle"] = model.predict(data)
        if feature_set == ('DensityRatio',):
            df["Pred_DensityRatio"] = model.predict(data)
        if feature_set == ('BBRatio',):
            df["Pred_BBRatio"] = model.predict(data)
        if feature_set == ('LengthRatio',):
            df["Pred_LengthRatio"] = model.predict(data)
        if feature_set == ('IntersectDistance',):
            df["Pred_IntersectDistance"] = model.predict(data)
        if feature_set == ('NumComponents',):
            df["Pred_NumComponents"] = model.predict(data)


        if feature_set == ('Circularity',):
            df["SVM_Contour_F"] = scale_to_binary(model.predict(data))
        if feature_set == ('Circularity', 'RemovedPoints'):
            df["SVM_Contour_AUC"] = scale_to_binary(model.predict(data))

        if feature_set == ('HandsAngle', 'BBRatio', 'LengthRatio', 'IntersectDistance', 'NumComponents'):
            df["SVM_Hands_F"] = scale_to_binary(model.predict(data))
        if feature_set == ('HandsAngle', 'DensityRatio', 'BBRatio', 'LengthRatio', 'IntersectDistance', 'NumComponents'):
            df["SVM_Hands_Acc"] = scale_to_binary(model.predict(data))

        if feature_set == ('DigitRadiusMean', 'DigitAngleStd', 'DigitAreaStd', 'ExtraDigits'):
            df["SVM_Digits_F"] = scale_to_binary(model.predict(data))
        if feature_set == ("DigitRadiusStd", "DigitAngleMean", "MissingDigits"):
            df["SVM_Digits_AUC"] = scale_to_binary(model.predict(data))
        if feature_set == ("DigitRadiusMean", "DigitRadiusStd", "DigitAngleStd", "DigitAreaMean",
                           "ExtraDigits", "MissingDigits"):
            df["SVM_Digits_Acc"] = scale_to_binary(model.predict(data))

        dummy_thresh = 0
        best_f1 = 0
        best_thresh = 0

        for i in range(100):
            threshed_predictions = apply_single_threshold(predictions, dummy_thresh)
            f1 = f1_score(test_labels, threshed_predictions, pos_label=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = dummy_thresh

            dummy_thresh += 0.01

        threshed_predictions = apply_single_threshold(predictions, best_thresh)
        accuracy = get_accuracy(test_labels, threshed_predictions)
        _, _, _, fnr = get_confusion(test_labels, threshed_predictions)

        average_precision = average_precision_score(test_labels, predictions, pos_label=0)
        auc = roc_auc_score(test_labels, predictions)
        rows_list.append({"Features": str(feature_set),
                          "F1 Score": best_f1,
                          "Accuracy": accuracy,
                          "FNR": fnr,
                          "Average Precision": average_precision,
                          "AUC Score": auc,
                          "Threshold": best_thresh})

    # Random guessing
    predictions = np.random.randint(2, size=len(test_labels))
    dummy_thresh = 0
    best_f1 = 0
    best_thresh = 0

    for i in range(100):
        threshed_predictions = apply_single_threshold(predictions, dummy_thresh)
        f1 = f1_score(test_labels, threshed_predictions, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(predictions, best_thresh)
    accuracy = get_accuracy(test_labels, threshed_predictions)
    _, _, _, fnr = get_confusion(test_labels, threshed_predictions)

    average_precision = average_precision_score(test_labels, predictions, pos_label=0)
    auc = roc_auc_score(test_labels, predictions)
    rows_list.append({"Features": "Random Guessing",
                      "F1 Score": best_f1,
                      "Accuracy": accuracy,
                      "FNR": fnr,
                      "Average Precision": average_precision,
                      "AUC Score": auc,
                      "Threshold": best_thresh})

    # Always guessing majority class
    predictions = np.ones((len(test_labels)))
    dummy_thresh = 0
    best_f1 = 0
    best_thresh = 0

    for i in range(100):
        threshed_predictions = apply_single_threshold(predictions, dummy_thresh)
        f1 = f1_score(test_labels, threshed_predictions, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = dummy_thresh

        dummy_thresh += 0.01

    threshed_predictions = apply_single_threshold(predictions, best_thresh)
    accuracy = get_accuracy(test_labels, threshed_predictions)
    _, _, _, fnr = get_confusion(test_labels, threshed_predictions)

    average_precision = average_precision_score(test_labels, predictions, pos_label=0)
    auc = roc_auc_score(test_labels, predictions)
    rows_list.append({"Features": "Assuming Majority",
                      "F1 Score": best_f1,
                      "Accuracy": accuracy,
                      "FNR": fnr,
                      "Average Precision": average_precision,
                      "AUC Score": auc,
                      "Threshold": best_thresh})

    return pd.DataFrame(rows_list)


########################################################################################################################

if __name__ == '__main__':
    data_file = "data/feature_data.csv"
    save_file = "data/SVM_" + data_file
    df = pd.read_csv(data_file)
    df = normalize(df)
    df.drop(df[df["LeftoverInk"] > 0.03].index, inplace=True)
    df = df.dropna(subset=["ClockContour", "ClockHands", "ClockNumbers", "Circularity", "RemovedPoints", "CenterDeviation",
                      "HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents",
                      "DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean",
                      "DigitAreaStd", "ExtraDigits", "MissingDigits"
                      ])

    columns = ["Circularity", "RemovedPoints", "CenterDeviation"]
    results_df = run_classifiers(df, columns, "ClockContour")
    results_df.to_csv("results/tables/svm_contour_results.csv")

    columns = ["HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents"]
    results_df = run_classifiers(df, columns, "ClockHands")
    results_df.to_csv("results/tables/svm_hands_results.csv")

    columns = ["DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits"]
    results_df = run_classifiers(df, columns, "ClockNumbers")
    results_df.to_csv("results/tables/svm_digit_results.csv")
    df.to_csv(save_file)