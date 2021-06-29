import numpy as np
import cv2

def generate_garbage():
    images = []
    background = np.zeros((112, 112))
    cv2.circle(background, (56, 56), 40, (255, 255, 255), 2)

    while len(images) < 3000:
        x = np.random.randint(0, 84)
        y = np.random.randint(0, 84)
        box = background[y:y+28, x:x+28]
        if np.sum(box) > 0:
            images.append(box)

    background = np.zeros((224, 224))
    cv2.circle(background, (112, 112), 80, (255, 255, 255), 2)
    while len(images) < 6000:
        x = np.random.randint(0, 196)
        y = np.random.randint(0, 196)
        box = background[y:y+28, x:x+28]
        if np.sum(box) > 0:
            images.append(box)

    dummy = np.zeros((28, 28))
    for i in range(1000):
        images.append(dummy)

    return np.array(images)

if __name__ == '__main__':
    generate_garbage()