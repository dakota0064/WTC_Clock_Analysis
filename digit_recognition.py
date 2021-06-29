from tensorflow.keras.models import load_model
from skimage import filters
import tensorflow as tf
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

from clock_digit_priors import get_priors

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def area_of_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return 0
    else:
        return w * h

def get_digits(img):

    model = load_model("augmented_mnist_digit_classifier.h5")
    intersect_threshold = 0.5
    box_threshold = 80
    number_threshold = 0.5
    sigma = 15
    recognized_digits = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    #----------------------------------------------------------------------------------------------------------------------#
    # Phase 0 - get the outer clock and the center point
    blurred = cv2.GaussianBlur(img.copy(), (5, 5), 0)
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

    #------------------------------------------------------------------------------------------------------------------#
    # # Phase 1 - No further preprocessing, use MSER to get blobs
    vis = img.copy()

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
                           _min_diversity=min_diversity, _max_evolution=max_evolution, _area_threshold=area_threshold,
                           _min_margin=min_margin, _edge_blur_size=edge_blur_size)
    regions, _ = mser.detectRegions(vis)
    boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    small_boxes = []
    for box in boxes:
        if box[2] > box_threshold or box[3] > box_threshold:
            continue
        else:
            small_boxes.append(box)

    #----------------------------------------------------------------------------------------------------------------------#
    # Phase 2 - Gaussian blurring and thresholding to solve for scanning abberations
    vis = img.copy()

    vis = cv2.GaussianBlur(vis, (3, 3), 0)
    _, vis = cv2.threshold(vis, 240, 255, cv2.THRESH_BINARY)

    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
                           _min_diversity=min_diversity, _max_evolution=max_evolution, _area_threshold=area_threshold,
                           _min_margin=min_margin, _edge_blur_size=edge_blur_size)
    regions, _ = mser.detectRegions(vis)
    boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    for box in boxes:
        if box[2] > box_threshold or box[3] > box_threshold:
            continue
        else:
            small_boxes.append(box)

    #----------------------------------------------------------------------------------------------------------------------#
    # Phase 3 - remove the outer contour and find boxes in what remains
    vis = img.copy()

    blurred = cv2.GaussianBlur(vis, (3, 3), 0)
    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
    inv = (255 - thresh)

    # draw the contour and center of the shape on the image
    if clock_contour is not None:
        cv2.drawContours(inv, [clock_contour], -1, (0, 0, 0), 17)

    #cv2.imshow("thr", inv)
    #cv2.waitKey(0)
    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
                           _min_diversity=min_diversity, _max_evolution=max_evolution, _area_threshold=area_threshold,
                           _min_margin=min_margin, _edge_blur_size=edge_blur_size)
    regions, _ = mser.detectRegions(inv)
    boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    for box in boxes:
        if box[2] > box_threshold or box[3] > box_threshold:
            continue
        else:
            small_boxes.append(box)

    #----------------------------------------------------------------------------------------------------------------------#
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

    #----------------------------------------------------------------------------------------------------------------------#
    # Phase 5 - remove intersecting boxes and feed remainder through classifier
    vis = img.copy()

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

        crop = vis[box[1]:box[1]+box[3]+1, box[0]:box[0]+box[2]+1]

        side_length = max(box[2], box[3]) + 6
        background = np.full((side_length, side_length), 255.0)

        x1 = int((side_length - box[2]) / 2)
        y1 = int((side_length - box[3]) / 2)
        x2 = x1 + box[2]+1
        y2 = y1 + box[3]+1

        background[y1:y2, x1:x2] = crop
        number_crops.append([cv2.resize(background, (28, 28)), box])

    for crop in number_crops:
        cropped_image = crop[0]
        box = crop[1]

        cropped_image = cropped_image.astype("float32") / 255
        cropped_image = np.expand_dims(cropped_image, 0)
        cropped_image = np.expand_dims(cropped_image, -1)
        probs = model.predict(cropped_image)
        box_center_x = box[0] + (box[2]/2)
        box_center_y = box[1] + (box[3]/2)

        angle = (-1 * (np.arctan2(box_center_y - center_Y, box_center_x - center_X) * 180 / np.pi) + 360) % 360
        #print(angle)
        priors = get_priors(angle, sigma)
        posteriors = (priors * probs)[0] / sum((priors * probs)[0])
        #print(priors)
        #print(probs)
        #print(posteriors)
        #print("")
        #plt.imshow(crop[0])
        #plt.show()
        #cv2.imshow("crop", crop[0])
        #cv2.waitKey(0)

        number = np.argmax(posteriors)
        if np.max(posteriors) > number_threshold:
            #print(np.max(posteriors))
            recognized_digits[number] += 1

    return img, recognized_digits

if __name__ == '__main__':

    base_directory = "cropped_images/"
    save_directory = "mser_and_connected/"
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory[:-1])

    for i, filename in enumerate(os.listdir(base_directory)):
        image = cv2.imread(base_directory + filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        labelled_image, digits = get_digits(gray)

        cv2.imwrite(save_directory + filename, labelled_image)
        if (i + 1) % 50 == 0:
            print("Labelled " + str(i + 1) + " images")

    # for number in recognized_digits.keys():
    #     recognized_digits[number] /= float(len(os.listdir(base_directory)))
    #
    # with open("digit_distribution.pkl", 'wb') as pkl_file:
    #     pickle.dump(recognized_digits, pkl_file)

