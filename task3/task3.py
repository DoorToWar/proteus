import cv2


def show_image(image, text):
    """Выводит изображение

    Использует библиотеку cv2 


    Args:
        image (ndarray): Массив с набором данных - представление каждого пикселя в формате чисел
        text (str): Название окна
    """
    cv2.imshow(text, image)  # выводит изображение
    cv2.waitKey(0)  # ждет ввода пользователя
    cv2.destroyAllWindows()  # закрывает окно


def print_info(image):
    """Выводит информацию об изображении (разрешение и само изображение в формате данных)

    Args:
        image (ndarray): Массив с набором данных - представление каждого пикселя в формате чисел
    """

    # вывод самого изображения (выводится набор данных - представление каждого пикселя в формате чисел)
    print("Изображение ", image)
    # вывод элементов shape (хранит высоту, ширину и кол-во каналов)
    print("Разрешение: ", image.shape[1], "x", image.shape[0])
    print("Количество каналов: ", image.shape[2])


def rotate(image, degree):
    """Поворачивает изображение

    Args:
        image (ndarray): Массив с набором данных - представление каждого пикселя в формате чисел
        degree (int): Угол, на который необходимо повернуть изображение
    """
    # получаем размер и изображения
    (h, w) = image.shape[:2]
    # находим центр
    center = (w/2, h/2)

    # создаётся матрица поворота вокруг центра
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    # применяется матрица поворота к изображению
    rotated_image = cv2.warpAffine(image, M, (w, h))
    show_image(rotated_image, str(degree))


def main():
    raw_image = cv2.imread(r"C:\proteus\task3\data\images\pcb.jpg",
                           cv2.IMREAD_COLOR)  # IMREAD_COLOR обеспечивает вывод изображения в цвете
    show_image(raw_image, "raw")

    print_info(raw_image)

    # изменение разрешения изображения
    resized_raw_image = cv2.resize(raw_image, (1920, 1080))
    print("Разрешение: ",
          resized_raw_image.shape[1], "x", resized_raw_image.shape[0])
    show_image(resized_raw_image, "resized raw")

    copy_resized_rotate = resized_raw_image

    # поворот изображения
    rotate(copy_resized_rotate, 45)
    rotate(copy_resized_rotate, 90)
    rotate(copy_resized_rotate, 180)

    copy_resized_mirror = resized_raw_image

    # отражение изображения
    flip_y = cv2.flip(copy_resized_mirror, 1)
    show_image(flip_y, "flip_y")
    flip_x = cv2.flip(copy_resized_mirror, 0)
    show_image(flip_x, "flip_x")

    # вырезается изображение размером 100x100
    cropped = resized_raw_image[500:600, 500:600]
    show_image(cropped, "crop")

    # получаем размер и изображения
    (h, w) = cropped.shape[:2]
    # находим центр
    (cropped_y, cropped_x) = (w//2, h//2)

    print("Center = ", (cropped_y, cropped_x))

    # изменение цвета центрального пикселя

    cropped[cropped_y, cropped_x] = [0, 0, 255]
    show_image(cropped, "red")

    # изменение цвета группы пикселей

    cropped[10:20, 10:20] = [255, 0, 0]
    show_image(cropped, "redes")

    # выделяется прямоугольник
    cv2.rectangle(cropped, (10, 10), (20, 20), (0, 0, 0), 2)
    show_image(cropped, "redes")

    # добавление текста на изображение
    cv2.putText(cropped, "rect", (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    show_image(cropped, "image_fin")

    # сохранение изображения
    cv2.imwrite(
        r"C:\GOG Games\proteus\task 2\proteus\proteus\task3\data\save\image_fin.jpg", cropped)


if __name__ == "__main__":
    main()
