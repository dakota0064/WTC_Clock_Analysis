import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import filters
from skimage.feature import hog
from sklearn.mixture import GaussianMixture
from get_contour_features import get_contour_score, smooth_contour

def determine_overlap(rect1, rect2):
    # Check if either rectangle is a line
    if (rect1[0] == rect1[2]) or (rect1[1] == rect1[3]) or (rect2[0] == rect2[2]) or (rect2[1] == rect2[3]):
        return False

    # If one rectangle is fully left of another, no intersection
    if(rect1[0] >= rect2[2] or rect2[0] >= rect1[2]):
        return False

    # If one rectangle is fully above another, no intersection
    if(rect1[1] >= rect2[3] or rect2[1] >= rect1[3]):
        return False

    return True

def get_maximum_bounding(rect1, rect2):
    x1, x2, y1, y2 = 0, 0, 0, 0
    if rect1[0] <= rect2[0]:
        x1 = rect1[0]
    else:
        x1 = rect2[0]

    if rect1[1] <= rect2[1]:
        y1 = rect1[1]
    else:
        y1 = rect2[1]

    if rect1[2] >= rect2[2]:
        x2 = rect1[2]
    else:
        x2 = rect2[2]

    if rect1[3] >= rect2[3]:
        y2 = rect1[3]
    else:
        y2 = rect2[3]

    return [x1, y1, x2, y2]

def get_hands_score(df, images, drawings):
    for i, image in enumerate(images):
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        edges = filters.sobel(thresh)

        low = 0.01
        high = 0.20

        hyst = filters.apply_hysteresis_threshold(edges, low, high).astype('uint8')
        hight = (edges > high).astype(np.uint8)
        inverted = (hight + hyst).astype(np.uint8)
        #inverted = cv2.dilate(inverted, (3, 3))

        #--------------------------------------------------------------------------------#
        # Get contour for the purposes of deleting #
        contours = cv2.findContours(inverted.copy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # loop over the contours
        biggest_moment = 0
        clock_contour = None

        for c in contours[0]:
            # compute the center of the contour
            M = cv2.moments(c)

            circle_center, radius = cv2.minEnclosingCircle(c)
            area = np.pi * (radius**2)

            if area > biggest_moment:
                biggest_moment = area
                clock_contour = c

        ###############################################
        # For testing features and visualizing images #
        ###############################################

        epsilon = 0.007 * cv2.arcLength(clock_contour, True)
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

        approx_area = cv2.contourArea(approx)
        approx_arc_length = cv2.arcLength(approx, True)
        approx_circularity = 4 * np.pi * approx_area / (approx_arc_length * approx_arc_length)

        for curve in (clock_contour, approx, hull):
            area = cv2.contourArea(curve)
            arc_length = cv2.arcLength(curve, True)
            circularity = 4 * np.pi * area / (arc_length * arc_length)
            if circularity > best_circularity:
                best_circularity = circularity
                best_curve = curve

        cv2.drawContours(inverted, [best_curve], -1, (0, 0, 0), 20)
        #--------------------------------------------------------------------------------------#

        # Get the connected components for the image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted)

        # Get the search box, determined using the radius of the minimum bounding circle.
        radius = df.at[i, "Radius"]
        cX, cY = df.at[i, "CenterPoint"][0]
        search_ratio = 0.3
        search_rect = [int(cX - radius * search_ratio), int(cY - radius * search_ratio),
                       int(cX + radius * search_ratio), int(cY + radius * search_ratio)]
        search_area = (radius * search_ratio) ** 2
        clock_area = np.pi * (radius**2)

        # Set a mask which will contain all connected components with pixels within the search box
        mask = np.zeros((vis.shape[0], vis.shape[1], 1), dtype='uint8')
        num_components = 0
        bounding_box = None

        for j in range(1, num_labels):
            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w = stats[j, cv2.CC_STAT_WIDTH]
            h = stats[j, cv2.CC_STAT_HEIGHT]
            component_rect = [x, y, x+w, y+h]
            component_area = w * h

            # Throw away large components, don't want the whole clock
            if component_area >= clock_area * .80:
                continue
            # Throw away small components, probably noise
            if component_area < 50:
                continue

            # Anything left which overlaps should be added to the mask
            if determine_overlap(component_rect, search_rect):
                component_mask = (labels==j).astype("uint8") * 255
                #cv2.imshow("vsi", inverted)
                #cv2.waitKey(0)
                mask = cv2.bitwise_or(mask, component_mask)
                num_components += 1
                if bounding_box == None:
                    bounding_box = [x, y, x+w, y+h]
                else:
                    bounding_box = get_maximum_bounding(bounding_box, [x, y, x+w, y+h])

        blank_ch = 255 * np.ones_like(mask)
        inv_mask = cv2.bitwise_not(mask)

        # Draw the hands onto the evaluation drawing in blue
        colored_mask = cv2.merge([blank_ch, inv_mask, inv_mask])
        drawings[i] = cv2.bitwise_and(drawings[i], colored_mask)

        # Use the black and white mask to determine hands features

        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        # Harris Corner detector parameters
        blockSize = 15
        apertureSize = 11
        k = 0.04
        threshold = 100

        kernel = np.ones((5, 5))
        fat_mask = mask
        for j in range(3):
            fat_mask = cv2.dilate(fat_mask, kernel)
        dst = cv2.cornerHarris(mask, blockSize, apertureSize, k)
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

        smallest_dist = 1000
        closest = (cX, cY)
        # Find the closest corner to the center
        for j in range(dst_norm.shape[0]):
            for k in range(dst_norm.shape[1]):
                if int(dst_norm[j, k]) > threshold:
                    distance = np.linalg.norm(np.array([k, j]) - np.array([cX, cY]))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest = (k, j)

        cv2.circle(drawings[i], (closest), 5, (255, 0, 155), -1)
        #cv2.imshow("Output", drawings[i])
        #cv2.waitKey(0)


        if np.any(np.where(mask > 0)):
            y_points, x_points = np.where(mask > 0)
            angles = (-1 * (np.arctan2(y_points - closest[1], x_points - closest[0]) * 180 / np.pi) + 360) % 360
            angles = angles.reshape(-1, 1)
            radii = np.linalg.norm(np.array([x_points, y_points]) - np.array([closest[0], closest[1]]).reshape(-1, 1),
                                   axis=0)

            mixture = GaussianMixture(n_components=2, random_state=0).fit(angles)
            mean1 = int(mixture.means_[0][0])
            mean2 = int(mixture.means_[1][0])

            # Get hands angle feature
            hands_angle = np.abs(mean1 - mean2)

            buffer = 7

            hand1_pts = len(np.where(((mean1 + buffer) > angles) & (angles > (mean1 - buffer)))[0])
            hand1_idxs = np.argwhere(((mean1 + buffer) > angles) & (angles > (mean1 - buffer)))
            try:
                hand1_radii = radii[hand1_idxs[:, 0]]
                hand1_length = np.max(hand1_radii)
            except:
                pass

            hand2_pts = len(np.where(((mean2 + buffer) > angles) & (angles > (mean2 - buffer)))[0])
            hand2_idxs = np.argwhere(((mean2 + buffer) > angles) & (angles > (mean2 - buffer)))
            try:
                hand2_radii = radii[hand2_idxs[:, 0]]
                hand2_length = np.max(hand2_radii)
            except:
                pass

            # Get hand length ratio feature
            try:
                short_hand = min(hand1_length, hand2_length)
                long_hand = max(hand1_length, hand2_length)
                length_ratio = short_hand / long_hand
            except:
                length_ratio = 0

            # Get hand density ratio feature
            little_hand = min(hand1_pts, hand2_pts)
            big_hand = max(hand1_pts, hand2_pts)
            density_ratio = little_hand / big_hand

            # Get bounding box ratio feature
            little_side = min(bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1])
            big_side = max(bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1])
            bb_ratio = little_side / big_side

            # print(mean1, hand1_pts)
            # print(mean2, hand2_pts)
            # print("")

            # Assign the features to the data frame
            df.at[i, "HandsAngle"] = hands_angle
            df.at[i, "DensityRatio"] = density_ratio
            df.at[i, "BBRatio"] = bb_ratio
            df.at[i, "LengthRatio"] = length_ratio
            df.at[i, "IntersectDistance"] = smallest_dist
            df.at[i, "NumComponents"] = num_components


        # descriptors, hog_image = hog(mask, orientations=24, pixels_per_cell=(8, 8),
        #                              cells_per_block=(1, 1), visualize=True)
        # descriptors = np.reshape(descriptors, (24, int(len(descriptors)/24)))
        # print(np.sum(descriptors, axis=1))
        #cv2.imshow("mask", hog_image)
        #cv2.waitKey(0)

    return df, drawings

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

    data["CenterPoint"] = None
    data["Circularity"] = 0.0
    data["RemovedPoints"] = 0
    data["Radius"] = 0.0
    data["CenterDeviation"] = 0.0

    data["HandsAngle"] = 0.0
    data["DensityRatio"] = 0.0
    data["BBRatio"] = 0.0
    data["LengthRatio"] = 0.0
    data["IntersectDistance"] = 0.0
    data["NumComponents"] = 0.0

    data = data.reset_index()
    data = data.drop(columns=['index'])

    data, drawings = get_contour_score(data, images)
    data, drawings = get_hands_score(data, images, drawings)