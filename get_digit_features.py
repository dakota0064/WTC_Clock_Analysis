import cv2
import os
import pandas as pd
import numpy as np
from skimage import filters
from scipy.stats import norm
from tensorflow.keras.models import load_model
from datetime import date, timedelta, datetime
from dateutil.parser import parse

from clock_digit_priors import get_angle_priors
from get_contour_features import get_contour_score
from get_hands_features import get_hands_score


def area_of_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return 0
    else:
        return w * h

########################################################################################################################


def get_digit_score(df, images, drawings):
    model = load_model("mnist_threshed_classifier.h5")
    intersect_threshold = 0.5
    box_threshold = 80
    number_threshold = 0.5
    sigma = 15
    for i, image in enumerate(images):
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        recognized_digits = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        clock_radius = df.at[i, "Radius"]

        # ----------------------------------------------------------------------------------------------------------------------#
        # Phase 0 - get the outer clock and the center point
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
        center_X = 0
        center_Y = 0
        for c in contours[0]:
            # compute the center of the contour
            M = cv2.moments(c)
            if M["m10"] > biggest_moment:
                biggest_moment = M["m10"]
                center_X = int(M["m10"] / M["m00"])
                center_Y = int(M["m01"] / M["m00"])

                clock_contour = c

        # ------------------------------------------------------------------------------------------------------------------#
        # # Phase 1 - No further preprocessing, use MSER to get blobs
        vis = image.copy()

        delta = 5
        min_area = 60
        max_area = 14400
        max_variation = 0.5
        min_diversity = 0.2
        max_evolution = 200
        area_threshold = 1.01
        min_margin = 0.003
        edge_blur_size = 5
        mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
                               _min_diversity=min_diversity, _max_evolution=max_evolution,
                               _area_threshold=area_threshold,
                               _min_margin=min_margin, _edge_blur_size=edge_blur_size)
        regions, _ = mser.detectRegions(gray)
        boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
        small_boxes = []
        for box in boxes:
            if box[2] > box_threshold or box[3] > box_threshold:
                continue
            else:
                small_boxes.append(box)

        # ----------------------------------------------------------------------------------------------------------------------#
        # Phase 2 - Gaussian blurring and thresholding to solve for scanning abberations
        vis = gray.copy()

        vis = cv2.GaussianBlur(vis, (3, 3), 0)
        _, vis = cv2.threshold(vis, 240, 255, cv2.THRESH_BINARY)

        mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
                               _min_diversity=min_diversity, _max_evolution=max_evolution,
                               _area_threshold=area_threshold,
                               _min_margin=min_margin, _edge_blur_size=edge_blur_size)
        regions, _ = mser.detectRegions(vis)
        boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
        for box in boxes:
            if box[2] > box_threshold or box[3] > box_threshold:
                continue
            else:
                small_boxes.append(box)

        # ----------------------------------------------------------------------------------------------------------------------#
        # Phase 3 - remove the outer contour and find boxes in what remains
        vis = gray.copy()

        blurred = cv2.GaussianBlur(vis, (3, 3), 0)
        thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
        inv = (255 - thresh)

        # draw the contour and center of the shape on the image
        if clock_contour is not None:
            cv2.drawContours(inv, [clock_contour], -1, (0, 0, 0), 17)

        # cv2.imshow("thr", inv)
        # cv2.waitKey(0)
        mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
                               _min_diversity=min_diversity, _max_evolution=max_evolution,
                               _area_threshold=area_threshold,
                               _min_margin=min_margin, _edge_blur_size=edge_blur_size)
        regions, _ = mser.detectRegions(inv)
        boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
        for box in boxes:
            if box[2] > box_threshold or box[3] > box_threshold:
                continue
            else:
                small_boxes.append(box)

        # ----------------------------------------------------------------------------------------------------------------------#
        # Phase 4 - find connected components with deleted contours
        # output = cv2.connectedComponentsWithStats(inverted.astype(np.uint8), 8, cv2.CV_32S)
        # #cv2.imshow("inv", inverted.astype(np.float32))
        # #cv2.waitKey(0)
        # (numLabels, labels, stats, centroid) = output
        #
        # for i in range(1, numLabels):
        #     x = stats[i, cv2.CC_STAT_LEFT]
        #     y = stats[i, cv2.CC_STAT_TOP]
        #     width = stats[i, cv2.CC_STAT_WIDTH]
        #     height = stats[i, cv2.CC_STAT_HEIGHT]
        #     if height > box_threshold or width > box_threshold:
        #         continue
        #     if height < 15 and width < 15:
        #         continue
        #     small_boxes.append((x, y, width, height))

        # ----------------------------------------------------------------------------------------------------------------------#
        # Phase 5 - remove intersecting boxes and feed remainder through classifier
        vis = gray.copy()
        blurred = cv2.GaussianBlur(vis, (3, 3), 0)
        thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]

        # Parse out boxes that cover the same area
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
                            if aoi / float(other_box_area) >= intersect_threshold:
                                small_boxes.remove(other_box)
                        else:
                            if aoi / float(box_area) >= intersect_threshold:
                                small_boxes.remove(box)
                    except:
                        pass

        small_boxes = list(set(small_boxes))
        number_crops = []
        for box in small_boxes:
            crop = thresh[box[1]:box[1] + box[3] + 1, box[0]:box[0] + box[2] + 1]

            side_length = max(box[2], box[3]) + 6
            background = np.full((side_length, side_length), 255.0)

            x1 = int((side_length - box[2]) / 2)
            y1 = int((side_length - box[3]) / 2)
            x2 = x1 + box[2] + 1
            y2 = y1 + box[3] + 1

            background[y1:y2, x1:x2] = crop
            number_crops.append([cv2.resize(background, (28, 28)), box])

        passable_crops = []
        radii = []
        angles = []
        for crop in number_crops:
            cropped_image = crop[0]
            box = crop[1]

            cropped_image = cropped_image.astype("float32") / 255
            cropped_image = np.expand_dims(cropped_image, 0)
            cropped_image = np.expand_dims(cropped_image, -1)
            probs = model.predict(cropped_image)
            box_center_x = box[0] + (box[2] / 2)
            box_center_y = box[1] + (box[3] / 2)

            radius = np.linalg.norm(np.array([center_X, center_Y]) - np.array([box_center_x, box_center_y]))
            # Disregard boxes with a center further than the clock radius
            if radius >= clock_radius:
                continue
            if radius <= 0.33 * clock_radius:
                continue
            r_ratio = radius / clock_radius
            angle = (-1 * (np.arctan2(box_center_y - center_Y, box_center_x - center_X) * 180 / np.pi) + 360) % 360
            average_r_ratio = 0.7

            angles.append(angle)
            radii.append(r_ratio)

            angle_priors = get_angle_priors(angle, sigma)
            dist_prob = norm.pdf(r_ratio, loc=average_r_ratio, scale=0.10)
            posteriors = dist_prob * ((angle_priors * probs)[0] / sum((angle_priors * probs)[0]))

            number = np.argmax(posteriors)
            # If garbage number is most probable, continue
            if number == 10:
                sorted = np.sort(posteriors)
                if sorted[-1] > sorted[-2] * 2:
                    continue
                else:
                    posteriors = posteriors[:-1]
                    number = np.argmax(posteriors)

            if np.max(posteriors) > number_threshold:
                # print(np.max(posteriors))
                recognized_digits[number] += 1
                passable_crops.append(crop)
                cv2.rectangle(drawings[i], (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 150, 0), 2)
                cv2.putText(drawings[i], str(number), (box[0] + 2, box[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)

        #cv2.imshow("drawing", drawings[i])
        #cv2.waitKey(0)
        df.at[i, "DigitRadiusMean"] = np.mean(radii)
        df.at[i, "DigitRadiusStd"] = np.std(radii)

    return df, drawings

#----------------------------------------------------------------------------------------------------------------------#
# Test function to make sure contour features are correct
if __name__ == '__main__':
    base_directory = "good_crops/"
    save_directory = "multi_feature/"
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory[:-1])

    df = pd.read_csv("clock_labels.csv")

    images = []
    files = []
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
            images.append(image)
            files.append(filename)


    df = df[df["found"] == 1]
    df = df.drop(columns=["found", 'visdt'])
    df = df.reset_index()
    print(df.head())
    print(df.tail())

    df["CenterPoint"] = None
    df["Circularity"] = 0.0
    df["RemovedPoints"] = 0.0
    df["Radius"] = 0.0
    df["CenterDeviation"] = 0.0

    df["HandsAngle"] = 0.0
    df["DensityRatio"] = 0.0
    df["BBRatio"] = 0.0
    df["LengthRatio"] = 0.0
    df["IntersectDistance"] = 0.0
    df["NumComponents"] = 0.0

    df["DigitRadiusMean"] = 0.0
    df["DigitRadiusStd"] = 0.0
    df["DigitAngleMean"] = 0.0
    df["DigitAngleStd"] = 0.0

    df, drawings = get_contour_score(df, images)
    df, drawings = get_hands_score(df, images, drawings)
    df, drawings = get_digit_score(df, images, drawings)

    print(df.head())
    print(df.tail())

    df.to_csv("multi_feature.csv")

    for i in range(len(drawings)):
        cv2.imwrite(save_directory + files[i], drawings[i])

        with open(save_directory + files[i][:-4] + "_features.txt", "w+") as txt_file:
            lines = []
            lines.append("Contour Features\n")
            lines.append("  Center Point: " + str(df.at[i, "CenterPoint"]) + '\n')
            lines.append("  Radius: " + str(df.at[i, "Radius"]) + '\n')
            lines.append("  Center Deviation: " + str(df.at[i, "CenterDeviation"]) + '\n')
            lines.append("  Circularity: " + str(df.at[i, "Circularity"]) + '\n')
            lines.append("  Removed Points: " + str(df.at[i, "RemovedPoints"]) + '\n')
            lines.append("" + '\n')
            lines.append("Hands Features" + '\n')
            lines.append("  Angle between hands: " + str(df.at[i, "HandsAngle"]) + '\n')
            lines.append("  Hands density ratio: " + str(df.at[i, "DensityRatio"]) + '\n')
            lines.append("  Hands length ratio: " + str(df.at[i, "LengthRatio"]) + '\n')
            lines.append("  Bounding box ratio: " + str(df.at[i, "BBRatio"]) + '\n')
            lines.append("  Intersection distance: " + str(df.at[i, "IntersectDistance"]) + '\n')
            lines.append("  Number of Components: " + str(df.at[i, "NumComponents"]) + '\n')

            txt_file.writelines(lines)
