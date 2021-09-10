from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

def normalize(df):
    df["CenterDeviation"] = df["CenterDeviation"] / np.max(df["CenterDeviation"])
    df["HandsAngle"] = df["HandsAngle"] / np.max(df["HandsAngle"])
    df["HandsAngle"] = np.where(df["HandsAngle"] > 180, df["HandsAngle"] - 180, df["HandsAngle"])
    df["IntersectDistance"] = df["IntersectDistance"] / np.max(df["IntersectDistance"])
    df["NumComponents"] = df["NumComponents"] / np.max(df["NumComponents"])
    df["DigitAngleMean"] = df["DigitAngleMean"] / np.max(df["DigitAngleMean"])
    df["DigitAngleStd"] = df["DigitAngleStd"] / np.max(df["DigitAngleStd"])
    df["DigitAreaMean"] = df["DigitAreaMean"] / np.max(df["DigitAreaMean"])
    df["DigitAreaStd"] = df["DigitAreaStd"] / np.max(df["DigitAreaStd"])
    df["ExtraDigits"] = df["ExtraDigits"] / np.max(df["ExtraDigits"])
    df["MissingDigits"] = df["MissingDigits"] / np.max(df["MissingDigits"])

    # Normalization based on feature analysis, to make monotonic
    df["DensityRatio"] = np.abs(0.37612489 - df["DensityRatio"])
    df["LengthRatio"] = np.abs(0.57538314 - df["LengthRatio"])
    df["BBRatio"] = np.abs(0.48631634 - df["BBRatio"])
    df["ExtraDigits"] = np.abs(0.30434783 - df["ExtraDigits"])
    df["DigitRadiusMean"] = np.abs(0.77046706 - df["DigitRadiusMean"])

    # Fixing "mesa" shaped behavior with 0 values
    df["HandsAngle"] = np.where(df["HandsAngle"] < 0.15, 1.0, df["HandsAngle"])
    df["NumComponents"] = np.where(df["NumComponents"] == 0.0, 1.0, df["NumComponents"])
    #df["DigitRadiusMean"] = np.where(df["DigitRadiusMean"] < 0.60, 1.0, df["DigitRadiusMean"])
    df["DigitAngleMean"] = np.where(df["DigitAngleMean"] < 0.05, 1.0, df["DigitAngleMean"])
    df["DigitAreaMean"] = np.where(df["DigitAreaMean"] == 0.0, 1.0, df["DigitAreaMean"])
    df["DigitAreaStd"] = np.where(df["DigitAreaStd"] == 0.0, 1.0, df["DigitAreaStd"])

    return df

########################################################################################################################

def run_classifiers(df, columns, label_name):

    labels = df[[label_name]].to_numpy()
    split_point = int(len(labels) * .8)

    train_labels = labels[:split_point]
    test_labels = labels[split_point:]

    rows_list = []
    model = RandomForestClassifier()#class_weight="balanced")
    feature_set = columns
    #print(feature_set)

    data = df[list(feature_set)].to_numpy()
    train_data = data[:split_point]
    test_data = data[split_point:]

    model.fit(train_data, train_labels.ravel())

    predictions = model.predict_proba(data)[:, 1]
    print(model.score(test_data, test_labels))
    df[label_name + "_pred"] = predictions

    return df


########################################################################################################################

if __name__ == '__main__':
    data_file = "data/feature_data.csv"
    save_file = "forest_mci.csv"
    df = pd.read_csv(data_file)
    #df = df.sample(frac=1).reset_index(drop=True)
    #df.to_csv(data_file)
    df = normalize(df)

    df["mci_26"] = np.where((df["Score"] >= 26), 1, 0)
    df["mci_25"] = np.where((df["Score"] >= 25), 1, 0)
    df["mci_24"] = np.where((df["Score"] >= 24), 1, 0)
    df["mci_23"] = np.where((df["Score"] >= 23), 1, 0)
    df["mci_22"] = np.where((df["Score"] >= 22), 1, 0)
    df["mci_21"] = np.where((df["Score"] >= 21), 1, 0)
    df["mci_19"] = np.where((df["Score"] >= 19), 1, 0)
    df["mci_18"] = np.where((df["Score"] >= 18), 1, 0)

    drop_columns = ["Circularity", "RemovedPoints", "CenterDeviation",
               "HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents",
               "DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits", "LeftoverInk", "PenPressure",
               "ClockContour", "ClockHands", "ClockNumbers", "Score"]

    columns = ["Circularity", "RemovedPoints", "CenterDeviation",
               "HandsAngle", "DensityRatio", "BBRatio", "LengthRatio", "IntersectDistance", "NumComponents",
               "DigitRadiusMean", "DigitRadiusStd", "DigitAngleMean", "DigitAngleStd", "DigitAreaMean", "DigitAreaStd",
               "ExtraDigits", "MissingDigits", "LeftoverInk", "PenPressure"]

    # Drop significantly uncircular contours, assume they are mistakes.
    df.drop(df[df["Circularity"] < 0.8].index, inplace=True)

    print(len(df))
    df.drop(df[df["LeftoverInk"] > 0.03].index, inplace=True)
    df.dropna(subset=drop_columns, inplace=True)

    print(len(df))
    results_df = run_classifiers(df, columns, "mci_26")
    results_df = run_classifiers(results_df, columns, "mci_25")
    results_df = run_classifiers(results_df, columns, "mci_24")
    results_df = run_classifiers(results_df, columns, "mci_23")
    results_df = run_classifiers(results_df, columns, "mci_22")
    results_df = run_classifiers(results_df, columns, "mci_21")
    results_df = run_classifiers(results_df, columns, "mci_19")
    results_df = run_classifiers(results_df, columns, "mci_18")
    results_df.to_csv(save_file)