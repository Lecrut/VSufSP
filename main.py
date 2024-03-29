import cv2 as cv
import numpy as np


def rescale(img, height=300):
    scale = height / img.shape[0]

    # Skaluj obraz
    resized_img = cv.resize(img, None, fx=scale, fy=scale)
    return resized_img


def create_potato_mask(image):
    image_temp = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    binr = cv.threshold(image_temp, 110, 255, cv.THRESH_BINARY)[1]
    binr = cv.GaussianBlur(binr, (9, 9), 0)

    invert = cv.bitwise_not(binr)

    contours, _ = cv.findContours(invert, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 3500]

    mask = np.zeros_like(image_temp)
    for cnt in filtered_contours:
        cv.drawContours(mask, [cnt], 0, 255, -1)  # wypełnij kontur wartością 1
        cv.drawContours(mask, [cnt], 0, 255, 1)  # narysuj obwód konturu wartością 1

    mask_image = np.copy(mask)
    mask = np.where(mask == 255, True, False)

    return mask_image, mask


def find_green_places(image, potato_mask):
    # Konwertuj obraz do przestrzeni kolorów HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Zdefiniuj kolor RGB
    rgb_color = np.uint8([[[137, 113, 56]]])

    # Konwertuj kolor do przestrzeni HSV
    hsv_color = cv.cvtColor(rgb_color, cv.COLOR_RGB2HSV)

    # Zdefiniuj zakres koloru w przestrzeni HSV
    lower_color = np.array([hsv_color[0][0][0], 100, 100])
    upper_color = np.array([hsv_color[0][0][0], 255, 255])

    # Utwórz maskę dla koloru
    mask = cv.inRange(hsv, lower_color, upper_color)
    mask_2 = potato_mask & mask

    # Znajdź kontury na masce
    contours, _ = cv.findContours(mask_2*255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Narysuj kontury na oryginalnym obrazie
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    return mask_2, image


def find_black_places(image, potato_mask):
    # Konwertuj obraz do przestrzeni kolorów HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Zdefiniuj kolor RGB
    lower_color = np.array([0, 0, 0])  # Zakres od czarnego
    upper_color = np.array([120, 120, 120])

    # Utwórz maskę dla koloru
    mask = cv.inRange(hsv, lower_color, upper_color)
    mask_2 = potato_mask & mask

    # Znajdź kontury na masce
    contours, _ = cv.findContours(mask_2*255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Narysuj kontury na oryginalnym obrazie
    cv.drawContours(image, contours, -1, (255, 0, 0), 3)

    return mask_2, image


def mark_defective_objects(image, main_mask, defect_mask, threshold=0.3):
    # Znajdź kontury na głównej masce
    contours, _ = cv.findContours(main_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    good = 0
    bad = 0
    # Przejrzyj wszystkie kontury
    for contour in contours:
        # Utwórz maskę dla pojedynczego obiektu
        single_object_mask = np.zeros_like(main_mask)
        cv.drawContours(single_object_mask, [contour], -1, (255), thickness=cv.FILLED)

        # Utwórz maskę defektów dla pojedynczego obiektu
        single_object_defect_mask = cv.bitwise_and(single_object_mask, defect_mask)

        # Oblicz udział maski defektu w masce pojedynczego obiektu
        defect_ratio = np.sum(single_object_defect_mask) / np.sum(single_object_mask)

        # Jeśli udział przekracza próg, zaznacz obiekt konturem na obrazie
        if defect_ratio * 1000 > threshold:
            cv.drawContours(image, [contour], -1, (0, 0, 255), 3)
            bad = bad + 1
        else:
            cv.drawContours(image, [contour], -1, (0, 255, 0), 3)
            good = good + 1

    return image, good, bad


def watching_potatoes():
    cap = cv.VideoCapture('images/20240109_162150 (online-video-cutter.com).mp4')
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

        image = frame

        if image.shape[0] > image.shape[1]:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        image_copy = image.copy()

        main_mask, image_mask = create_potato_mask(image)
        green_mask, image = find_green_places(image, image_mask)
        black_mask, image = find_black_places(image, image_mask)

        # cv.imshow('first', image)

        # cv.imshow('black', black_mask*255)
        # cv.imshow('green', green_mask*255)

        # defect_mask = (green_mask | black_mask) * 255

        # cv.imshow('compilation', defect_mask)
        # cv.imshow('main', main_mask)

        # maska1 = main_mask
        # maska2 = defect_mask
        #
        # maska1_kolor = cv.cvtColor(maska1, cv.COLOR_GRAY2BGR)
        # maska1_kolor[maska1 != 0] = [255, 255, 255]
        #
        # cv.imshow('main', maska1_kolor)
        #
        # # Zmień kolor maski2 na inny kolor, na przykład na czerwony
        # maska2_kolor = cv.cvtColor(maska2, cv.COLOR_GRAY2BGR)
        # maska2_kolor[maska2 != 0] = [0, 0, 255]  # czerwony kolor
        #
        # # Nałóż maska1 na maska2_kolor
        # finalna_maska = cv.addWeighted(maska1_kolor, 0.5, maska2_kolor, 0.5, 0)
        #
        # # Wyświetl finalną maskę
        # cv.imshow('Finalna Maska', finalna_maska)

        image_copy, good, bad = mark_defective_objects(image_copy, main_mask, (green_mask | black_mask))

        image_copy = rescale(image_copy, 500)

        cv.putText(image_copy, 'Liczba dobrych ziemniakow: {}'.format(good), (10, 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv.putText(image_copy, 'Liczba zlych ziemniakow: {}'.format(bad), (10, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow('second', image_copy)
        cv.waitKey(2)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    watching_potatoes()
