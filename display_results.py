import pickle
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np

def display_digit_distribution():
    with open('digit_distribution.pkl', 'rb') as pkl_file:
        recognized_digits = dict(pickle.load(pkl_file))

    expected_digits = {0:1, 1:5, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}

    x = []
    height = []
    labels = []

    x2 = []
    height2 = []
    for i in range(len(recognized_digits.keys())):
        x.append(i)
        height.append(recognized_digits[i])
        labels.append(str(i))

        x2.append(i)
        height2.append(expected_digits[i])

    plt.figure(1, figsize=(10, 5))
    axes = plt.gca()
    axes.set_ylim([0, 5])

    plt.subplot(1, 2, 1)
    plt.bar(x, height, color="blue", tick_label = labels)
    plt.title("Average Number of digits per clock")
    plt.xlabel("Digit")
    plt.ylabel("Average num of occurences")
    axes = plt.gca()
    axes.set_ylim([0, 7])


    plt.subplot(1, 2, 2)
    plt.bar(x2, height2, color="red", tick_label = labels)
    plt.title("Expected Number of digits per clock")
    plt.xlabel("Digit")
    plt.ylabel("Expected num of occurences")
    axes = plt.gca()
    axes.set_ylim([0, 7])

    plt.show()


def display_pearson_heatmap():
    plt.figure(1, (10, 10))
    plt.suptitle("WTC Data Correlation Map", fontsize=24)
    data = pd.read_csv("multi_feature.csv")
    contours = data[['ClockContour', 'Circularity', 'Radius', 'CenterDeviation', 'RemovedPoints']]
    hands = data[["ClockHands", 'HandsAngle', 'DensityRatio', 'BBRatio', 'LengthRatio', 'IntersectDistance', 'NumComponents']]

    matrix = np.tril(hands.corr())
    seaborn.heatmap(hands.corr(), fmt='g', cmap='coolwarm', mask=matrix)
    plt.savefig("small_data_heatmap")
    plt.show()

if __name__ == '__main__':
    display_pearson_heatmap()
