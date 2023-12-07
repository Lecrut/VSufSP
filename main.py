import cv2 as cv
import numpy as np
from collections import Counter

# todo: wyeliminować tło


def rescale(img, percentage=100):
    return cv.resize(img, None, fx=percentage/100, fy=percentage/100)


def find_black_places(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, threshold = cv.threshold(gray, 30, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image


# def get_most_common_color(img):
#     flattened = img.reshape(-1, img.shape[-1])
#
#     color, count = Counter(map(tuple, flattened)).most_common(1)[0]
#     print(color)
#     return color


def delete_background(image):
    binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # define the kernel
    kernel = np.ones((3, 3), np.uint8)

    # invert the image
    invert = cv.bitwise_not(binr)

    # use morph gradient
    return cv.morphologyEx(invert, cv.MORPH_GRADIENT, kernel)


def watching_potatoes():
    cap = cv.VideoCapture('images/mandarynka_TEST.jpg')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()

    image = rescale(frame, 45)

    if image.shape[0] > image.shape[1]:
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

    # image = find_black_places(image)
    a, b, c = cv.split(image)
    image = delete_background(a)
    while True:
        cv.imshow('frame', image)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    watching_potatoes()
