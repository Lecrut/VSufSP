import cv2
import cv2 as cv
import numpy as np

# todo: wyeliminować tło


def rescale(img, precentage = 100):
    return cv2.resize(img, None, fx=precentage/100, fy=precentage/100)


def watching_potatoes():
    cap = cv.VideoCapture('images/ziemniak_dobry.jpg')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()

    # todo: do usunięcia / napisania warunek żeby zdj były poziome
    image = rescale(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 50)

    while True:
        cv.imshow('frame', image)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    watching_potatoes()
