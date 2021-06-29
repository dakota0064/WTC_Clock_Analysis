import pandas as pd
import cv2
import numpy as np
import os

from datetime import date, timedelta, datetime
from dateutil.parser import parse
from skimage import filters
from digit_recognition import get_digits

def load_images(base_directory, df):

    images = []
    df['found'] = 0
    df["filename"] = ""

    # Fill in date column with desired values
    base_date = date(1960, 1, 1)
    df["date"] = None
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('date')] = base_date + timedelta(int(df.iloc[i]["visdt"]))

    # Loop through and mark out which rows we actually have images for
    for filename in os.listdir(base_directory):
        names = filename.split(" ")
        if "n.d" in names[-1]:
            continue
        evaluation_date = parse(names[-1][:-4]).date()
        conditions = [df['mrn'].eq(names[0]) & df['date'].eq(evaluation_date)]
        choices = [1]
        df['found'] = np.select(conditions, choices, default=df['found'])

        # Load the images if we have a row for it
        if np.where(df['mrn'].eq(names[0]) & df['date'].eq(evaluation_date), True, False).any():
            df['filename'] = np.select(conditions, [filename], default=df['filename'])
            image = cv2.imread(base_directory + filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(img)


    df = df[df["found"] == 1]
    df = df.drop(columns=["found", 'visdt'])

    df["CenterPoint"] = None
    df["Circularity"] = 0.0
    df["HandsRatio"] = 0.0
    df["Digits"] = 0.0
    return images, df

#----------------------------------------------------------------------------------------------------------------------#


def get_contour_score(df, images):
    for i, image in enumerate(images):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1]
        edges = filters.sobel(thresh)

        low = 0.01
        high = 0.20

        hyst = filters.apply_hysteresis_threshold(edges, low, high).astype(int)
        hight = (edges > high).astype(np.uint8)
        inverted = (hight + hyst)

        contours = cv2.findContours(inverted.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # loop over the contours
        biggest_moment = 0
        clock_contour = None
        cX = 0
        cY = 0
        circularity = 1
        for c in contours[0]:
            # compute the center of the contour
            M = cv2.moments(c)
            if M["m10"] > biggest_moment:
                biggest_moment = M["m10"]
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                area = cv2.contourArea(c)
                arc_length = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area / (arc_length * arc_length)
                clock_contour = c

        df.at[i, "Circularity"] = circularity
        df.at[i, "CenterPoint"] = (cX, cY)

    return df

#----------------------------------------------------------------------------------------------------------------------#


def get_hands_score(df, images):

    for img_index, image in enumerate(images):

        cX, cY = df.iloc[img_index, df.columns.get_loc('CenterPoint')]

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1]
        edges = filters.sobel(thresh)

        low = 0.01
        high = 0.20

        hyst = filters.apply_hysteresis_threshold(edges, low, high).astype(int)
        hight = (edges > high).astype(np.uint8)
        inverted = (hight + hyst)

        output = cv2.connectedComponentsWithStats(inverted.astype(np.uint8), 8, cv2.CV_32S)
        (numLabels, labels, stats, centroid) = output

        largest_area = 0
        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                label = i

        closest_distance = 10000
        width = 0
        height = 0

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            center_x = x + (w / 2)
            lower_y = y + h

            if area < 200 or (area / largest_area) > 0.7 or area > 3000 or i == label:
                continue
            distance = np.linalg.norm(np.array([cX, cY]) - np.array([center_x, lower_y]))
            if distance - (area * 0.01) < closest_distance - (largest_area * 0.01):
                closest_distance = distance
                width = w
                height = h

            big_side = max(width, height)
            small_side = min(width, height)
            ratio = small_side / big_side

        df.at[img_index, "HandsRatio"] = ratio

    return df

#----------------------------------------------------------------------------------------------------------------------#


def get_digits_score(df, images):

    expected_digits = {0: 1, 1: 5, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

    for img_index, image in enumerate(images):

        _, recognized_digits = get_digits(image)

        missing_digits = 0

        for digit in recognized_digits.keys():
            missing = expected_digits[digit] - recognized_digits[digit]
            missing = max(missing, 0)
            missing_digits += missing

        df.at[img_index, "Digits"] = missing_digits

    return df

#----------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':

    data = pd.read_csv("clock_labels.csv")
    data["ClockTotal"] = data["ClockContour"] + data["ClockNumbers"] + data["ClockHands"]
    print(data.tail())

    images, data = load_images("good_crops/", data)
    print("Loaded ", len(images), " images")
    data = data.reset_index()
    data = data.drop(columns=['index'])
    print(data.tail())

    data = get_contour_score(data, images)
    print(data.tail())

    data = get_hands_score(data, images)
    print(data.head())

    data = get_digits_score(data, images)
    print(data.head())

    data.to_csv("features_and_labels.csv")

