import cv2 as cv
import numpy as np
from collections import Counter

# todo: wyeliminować tło


def rescale(img, percentage=100):
    return cv.resize(img, None, fx=percentage/100, fy=percentage/100)


def find_black_places(image, image_contours):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # im wiekszy tym wiecej plamek znajdzie
    threshold_value = 60
    _, threshold = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (255, 0, 0), 3)

    contours, _ = cv.findContours(image_contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 0, 255), 3)

    return image


def remove_background_outside_contour(image):
    # Konwertuj obraz do skali szarości
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Zastosuj binaryzację, aby uzyskać binaryzny obraz
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Znajdź kontury na binaryznym obrazie
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Stwórz pustą maskę
    mask = np.zeros_like(image)

    # Narysuj kontury na masce
    cv.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv.FILLED)

    # Nałóż maskę na oryginalny obraz
    return cv.bitwise_and(image, mask)


def count_objects(image):
    # Konwertuj obraz do skali szarości
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Zastosuj binaryzację, aby uzyskać binaryzny obraz
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Znajdź kontury na binaryznym obrazie
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Zlicz obiekty na podstawie liczby konturów
    object_count = len(contours)

    return object_count


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
    cap = cv.VideoCapture('images/3_ziemniaki.jpg')
    # cap.set(cv.CAP_PROP_POS_FRAMES, 10)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # liczba klatek na sekundę
    # cap.set(cv.CAP_PROP_FPS, 10)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        image = rescale(frame, 45)

        if image.shape[0] > image.shape[1]:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        a, b, c = cv.split(image)
        image_contours = delete_background(a)

        image = find_black_places(image, image_contours)
        # image = remove_background_outside_contour(image)

        cv.imshow('frame', image)
        print(count_objects(image))
        cv.waitKey(300000)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    watching_potatoes()
