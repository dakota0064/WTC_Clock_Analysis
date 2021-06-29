import cv2
import numpy as np
import os
from skimage import filters
import matplotlib.pyplot as plt

def find_centroids(base_directory, save_directory=None):
    for filename in os.listdir(base_directory):

        #if filename != "W03516 05-30-2014.jpg":
        #    continue

        image = cv2.imread(base_directory + filename)
        image = image[65:, :]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #inverted = (255 - gray)
        #blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        #blurred = cv2.GaussianBlur(blurred, (15, 15), 0)
        #thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1]
        #inverted = (255 - thresh)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1]
        edges = filters.sobel(thresh)

        low = 0.01
        high = 0.20

        hyst = filters.apply_hysteresis_threshold(edges, low, high).astype(int)
        hight = (edges > high).astype(np.uint8)
        inverted = (hight + hyst)

        #cv2.imshow("thresh", inverted)
        #cv2.waitKey(0)

        contours = cv2.findContours(inverted.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # loop over the contours
        biggest_moment = 0
        clock_contour = None
        cX = 0
        cY = 0
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

        print(filename, circularity)
        cv2.drawContours(image, clock_contour, -1, (0, 0, 255), thickness=5)



        #cv2.imshow("img", inverted.astype(np.float32))
        #cv2.waitKey(0)

        output = cv2.connectedComponentsWithStats(inverted.astype(np.uint8), 8, cv2.CV_32S)
        (numLabels, labels, stats, centroid) = output

        # Compute the centroid by taking the average of the centroids of all unique cc's
        #center = np.mean(centroid, axis=0)
        #cX = int(center[0])
        #cY = int(center[1])


        # Phase 1 - find the largest connected component and record its center
        largest_area = 0

        for i in range(1, numLabels):
             area = stats[i, cv2.CC_STAT_AREA]
             if area > largest_area:
                 largest_area = area
                 cX2, cY2 = centroid[i]
                 label = i
        #
        # cX2 = int(cX2)
        # cY2 = int(cY2)
        #
        # image_center_x = int(image.shape[1] / 2)
        # image_center_y = int(image.shape[0] / 2)
        # distance_1 = np.linalg.norm(np.array([image_center_x, image_center_y]) - np.array([cX1, cY1]))
        # distance_2 = np.linalg.norm(np.array([image_center_x, image_center_y]) - np.array([cX2, cY2]))
        # if distance_1 < distance_2:
        #     cX = cX1
        #     cY = cY1
        # else:
        #     cX = cX2
        #     cY = cY2

        cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        # Phase 2 - find the next largest connected component closest to the center

        closest_distance = 10000
        largest = 0
        closest_x = 0
        closest_y = 0
        width = 0
        height = 0

        # for i in range(1, numLabels):
        #     area = stats[i, cv2.CC_STAT_AREA]
        #     if i == label or area > 2000:
        #         continue
        #     if area > largest and (area / largest_area) < 0.6:
        #         largest = area
        #         closest_x = stats[i, cv2.CC_STAT_LEFT]
        #         closest_y = stats[i, cv2.CC_STAT_TOP]
        #         width = stats[i, cv2.CC_STAT_WIDTH]
        #         height = stats[i, cv2.CC_STAT_HEIGHT]

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            center_x = x + (w/2)
            lower_y = y + h

            if area < 200 or (area / largest_area) > 0.7 or area > 3000:
                continue
            distance = np.linalg.norm(np.array([cX, cY]) - np.array([center_x, lower_y]))
            if distance - (area * 0.01) < closest_distance - (largest_area * 0.01):
                closest_distance = distance
                closest_x = x
                closest_y = y
                width = w
                height = h
                #print(area)

        cv2.rectangle(image, (closest_x, closest_y), (closest_x+width, closest_y+height), (255, 0, 0), 2)

        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        cv2.imwrite(save_directory + filename, image)
        if (i+1) % 50 == 0:
            print("Labelled " + str(i+1) + " images")

if __name__ == '__main__':
    save_directory = "hands_and_contour/"
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory[:-1])
    base_directory = "cropped_images/"

    find_centroids(base_directory, save_directory=save_directory)