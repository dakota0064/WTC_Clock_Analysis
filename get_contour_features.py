import pandas as pd
import cv2
import numpy as np
import os

from skimage import filters


def smooth_contour(points_list):
    for i, point1 in enumerate(points_list):
        # Restart if unstable, point has been deleted

        # Get adjacent points
        if i - 1 < 0:
            left = -1
        else:
            left = i - 1

        if i + 1 >= len(points_list):
            right = 0
        else:
            right = i + 1

        left_dist = np.linalg.norm(points_list[i] - points_list[left])
        right_dist = np.linalg.norm(points_list[i] - points_list[right])

        if left_dist < right_dist:
            min_dist = left_dist
            closest_point = left
        else:
            min_dist = right_dist
            closest_point = right
        for j, point2 in enumerate(points_list):
            # Restart full while loop if unstable, point has been deleted
            if i == j:
                continue
            # See if there is a closer point, if so remove previous closest point and restart
            if np.linalg.norm(points_list[i] - points_list[j]) < min_dist and j != closest_point:
                reduced_points = np.delete(points_list, [closest_point], axis=0)
                return reduced_points, False

    return points_list, True

#----------------------------------------------------------------------------------------------------------------------#

def get_contour_score(df, images):
    drawn_images = []
    for i, image in enumerate(images):
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1]
        edges = filters.sobel(thresh)

        low = 0.01
        high = 0.20

        hyst = filters.apply_hysteresis_threshold(edges, low, high).astype(int)
        hight = (edges > high).astype(np.uint8)
        inverted = (hight + hyst)

        contours = cv2.findContours(inverted.copy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # loop over the contours
        biggest_moment = 0
        clock_contour = None

        for c in contours[0]:
            # compute the center of the contour
            M = cv2.moments(c)

            circle_center, radius = cv2.minEnclosingCircle(c)
            area = np.pi * (radius**2)
            arc_length = cv2.arcLength(c, True)
            circularity = 4 * np.pi * cv2.contourArea(c) / (arc_length * arc_length)

            if area > biggest_moment and circularity > 0.1:
                biggest_moment = area
                clock_contour = c

        ###############################################
        # For testing features and visualizing images #
        ###############################################

        epsilon = 0.009 * cv2.arcLength(clock_contour, True)
        hull = cv2.convexHull(clock_contour, returnPoints=True)
        approx = cv2.approxPolyDP(clock_contour, epsilon, True)

        removals = 0
        while True:
            approx, stable = smooth_contour(approx)
            if stable:
                break
            else:
                removals += 1

        # At this point we have 3 approximations of the contour; original, reduced, and hull

        best_curve = None
        best_circularity = 0
        biggest_area = 0
        best_compound = 0.0

        approx_area = cv2.contourArea(approx)
        approx_arc_length = cv2.arcLength(approx, True)
        approx_circularity = 4 * np.pi * approx_area / (approx_arc_length * approx_arc_length)

        for curve in (clock_contour, approx, hull):
            area = cv2.contourArea(curve)
            arc_length = cv2.arcLength(curve, True)
            circularity = 4 * np.pi * area / (arc_length * arc_length)
            if (0.5 * circularity) + (0.5 * area) > best_compound:
                best_compound = (0.5 * circularity) + (0.5 * area)
                #best_circularity = circularity
                best_curve = curve

        M = cv2.moments(best_curve)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.drawContours(vis, [best_curve], -1, (0, 0, 255), 2)
        circle_center, radius = cv2.minEnclosingCircle(best_curve)

        cv2.circle(vis, (cX, cY), 5, (0, 0, 255), -1)
        #cv2.putText(image, "real center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.circle(vis, (int(circle_center[0]), int(circle_center[1])), int(radius), (0, 255, 255), 2)
        cv2.circle(vis, (int(circle_center[0]), int(circle_center[1])), 5, (0, 255, 255), -1)
        #cv2.putText(vis, "bounding center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        center_deviation = np.linalg.norm(circle_center - np.array([cX, cY]))
        #cv2.drawContours(vis, [approx], -1, (255, 255, 0), 2)
        #cv2.drawContours(vis, [hull], -1, (0, 0, 255), 2)

        #cv2.drawContours(vis, clock_contour, -1, (0, 0, 255), thickness=5)
        if best_circularity != approx_circularity:
            removals = 0

        # print("Cicularity: (Arc Length)", circularity)
        # print("Removed Points: ", removals)
        # print("Center Deviation: ", center_deviation)
        # print(cX, cY)
        # print("")

        df.at[i, "Circularity"] = circularity
        df.at[i, "CenterPoint"] = [(cX, cY)]
        df.at[i, "RemovedPoints"] = removals
        df.at[i, "Radius"] = radius
        df.at[i, "CenterDeviation"] = center_deviation

        #cv2.imshow("contour features", vis)
        #cv2.waitKey(0)
        drawn_images.append(vis)

    return df, drawn_images


# Test function to make sure contour features are correct
if __name__ == '__main__':

    data = pd.read_csv("small_sample_labels.csv")
    data["ClockTotal"] = data["ClockContour"] + data["ClockNumbers"] + data["ClockHands"]

    good_image = cv2.imread("good_crops/W00623 n.d..jpg")
    bad_image = cv2.imread("good_crops/W11692 07-14-2014.jpg")
    worse_image = cv2.imread("good_crops/W07059 03-20-2014.jpg")
    images = []
    images.append(good_image)
    images.append(bad_image)
    images.append(worse_image)

    data["CenterPoint"] = [(0, 0)]
    data["Circularity"] = 0.0
    data["RemovedPoints"] = 0
    data["Radius"] = 0.0
    data["CenterDeviation"] = 0.0

    data = data.reset_index()
    data = data.drop(columns=['index'])
    print(data.tail())

    data, images = get_contour_score(data, images)
    print(data.tail())