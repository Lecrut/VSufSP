import cv2 as cv
import numpy as np

# todo: wyeliminować tło


def rescale(img, percentage=100):
    return cv.resize(img, None, fx=percentage/100, fy=percentage/100)


def find_black_places(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, threshold = cv.threshold(gray, 30, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image


def watching_potatoes():
    cap = cv.VideoCapture('images/ziemniak_zgnily2.jpg')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()

    # todo: do usunięcia / napisania warunek żeby zdj były poziome
    image = rescale(cv.rotate(frame, cv.ROTATE_90_CLOCKWISE), 50)
    image = find_black_places(image)

    while True:
        cv.imshow('frame', image)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def find_black_places(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, threshold = cv.threshold(gray, 30, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image

if __name__ == '__main__':
    watching_potatoes()
