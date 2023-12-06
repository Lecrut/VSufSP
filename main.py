import cv2 as cv
import numpy as np

# todo: zmniejszyć zdjęcia o połowę minimum
# todo: wyeliminować tło


def watching_potatoes():
    cap = cv.VideoCapture('images/ziemniak_dobry.jpg')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()

    while True:
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    watching_potatoes()
