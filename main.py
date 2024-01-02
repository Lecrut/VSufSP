import cv2 as cv
import numpy as np


def rescale(img, height=200):
    scale = height / img.shape[0]

    # Skaluj obraz
    resized_img = cv.resize(img, None, fx=scale, fy=scale)
    return resized_img


def find_green_places(image):
    # Konwertuj obraz do przestrzeni kolorów HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Zdefiniuj kolor RGB
    rgb_color = np.uint8([[[137, 113, 56]]])

    # Konwertuj kolor do przestrzeni HSV
    hsv_color = cv.cvtColor(rgb_color, cv.COLOR_RGB2HSV)

    # Zdefiniuj zakres koloru w przestrzeni HSV
    #todo: ustawić dla finalnego obrazu tą tolerację

    # lower_color = np.array([hsv_color[0][0][0] - 10, 100, 100])
    # upper_color = np.array([hsv_color[0][0][0] + 10, 255, 255])

    lower_color = np.array([hsv_color[0][0][0], 100, 100])
    upper_color = np.array([hsv_color[0][0][0], 255, 255])

    # Utwórz maskę dla koloru
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Znajdź kontury na masce
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Narysuj kontury na oryginalnym obrazie
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    return image


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
    threshold_value = 60
    _, binary = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY)

    # Znajdź kontury na binaryznym obrazie
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Stwórz pustą maskę
    mask = np.zeros_like(image)

    # Narysuj kontury na masce
    cv.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv.FILLED)

    # Nałóż maskę na oryginalny obraz
    return cv.bitwise_and(image, mask)


# todo: naprawic
def count_objects(image):
    # Konwertuj obraz do skali szarości
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Zastosuj binaryzację, aby uzyskać binaryzny obraz
    threshold_value = 60
    _, binary = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY)

    # Znajdź kontury na binaryznym obrazie
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Zlicz obiekty na podstawie liczby konturów
    object_count = len(contours)

    return object_count

# todo: rozkminic po co i czy potrzebne
# def get_most_common_color(img):
#     flattened = img.reshape(-1, img.shape[-1])
#
#     color, count = Counter(map(tuple, flattened)).most_common(1)[0]
#     print(color)
#     return color

# todo: do poprawy
def delete_background(image):
    binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # define the kernel
    kernel = np.ones((3, 3), np.uint8)

    # invert the image
    invert = cv.bitwise_not(binr)

    # use morph gradient
    return cv.morphologyEx(invert, cv.MORPH_GRADIENT, kernel)


def watching_potatoes():
    cap = cv.VideoCapture('images/nagranie_1.mp4')
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

        image = rescale(frame, 600)

        if image.shape[0] > image.shape[1]:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        a, b, c = cv.split(image)
        image_contours = delete_background(a)

        image = find_green_places(image)
        image = find_black_places(image, image_contours)
        # image = remove_background_outside_contour(image)

        cv.imshow('frame', image)
        print(count_objects(image))
        cv.waitKey(5)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    watching_potatoes()
