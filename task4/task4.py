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


def main():
    raw_image


if __name__ == "__main__":
    main()
