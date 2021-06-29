import cv2
import numpy as np
import matplotlib.pyplot as plt

def area_of_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return 0
    else:
        return w * h

img = cv2.imread("cropped_images/W00088 04-11-2014.jpg", 0)
vis = img.copy()
vis = vis[70:, :]
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(vis)
boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
small_boxes = []
for box in boxes:
    if box[2] > 80 or box[3] > 80:
        continue
    else:
        small_boxes.append(box)

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
                    small_boxes.remove(other_box)
                else:
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
    number_crops.append(cv2.resize(background, (28, 28)))

for crop in number_crops:
    plt.imshow(crop)
    plt.show()