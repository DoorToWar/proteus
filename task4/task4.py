import cv2
import numpy as np


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


def BGR2CMYK(image):
    """Преобразовние из BGR в CMYK

    Args:
        image (ndarray): Массив с набором данных - представление каждого пикселя в формате чисел

    Returns:
        fin_img (ndarray): Массив с набором данных - представление каждого пикселя в формате чисел
    """

    colours = image.astype(float)/255
    b, g, r = cv2.split(colours)  # изображение делится на каналы

    # расчет кадого параметра
    k = 1 - np.max([b, g, r], axis=0)
    c = (1 - r - k) / (1-k)
    m = (1 - g - k) / (1-k)
    y = (1 - b - k) / (1-k)
    # с помощью merge "собираем" изображение
    fin_img = cv2.merge([c, m, y, k])

    # для корректного отображения значение приводится обратно в диапозон от 0 до 255
    fin_img = (fin_img * 255).astype(np.uint8)
    return (fin_img)


def math_with_chanel(chanel):
    """Тестирование базовых математических операций над каналом

    Args:
        chanel (ndarray): Канал изображения
    """
    ch_plus = chanel + 15
    ch_minus = chanel - 15
    ch_mult = chanel * 2
    ch_div = chanel / 60
    show_image(chanel, "chanel b4 +")
    show_image(ch_plus, "chanel after +")
    show_image(chanel, "chanel b4 *")
    show_image(ch_mult, "chanel after *")
    show_image(chanel, "chanel b4 -")
    show_image(ch_minus, "chanel after -")
    show_image(chanel, "chanel b4 /")
    show_image(ch_div, "chanel after /")
    ch_div = ch_div.astype(np.uint8)

    return (ch_plus, ch_mult, ch_minus, ch_div)


def main():
    raw_image = cv2.resize(cv2.imread(
        r"C:\proteus\task4\pcb.jpg", cv2.IMREAD_COLOR), (1920, 1080))
    show_image(raw_image, "raw")

    # Преобразование из BGR в RGB
    rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    show_image(rgb, "rgb")

    # Преобразование из BGR в XYZ
    xyz = cv2.cvtColor(raw_image, cv2.COLOR_BGR2XYZ)
    show_image(xyz, "xyz")
    # Преобразование из BGR в HSV
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    show_image(hsv, "hsv")
    # Преобразование из BGR в HLS
    hls = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HLS)
    show_image(hls, "hls")
    # Преобразование из BGR в LAB
    lab = cv2.cvtColor(raw_image, cv2.COLOR_BGR2Lab)
    show_image(lab, "lab")
    # Преобразование из BGR в LUV
    luv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2Luv)
    show_image(luv, "luv")
    # Преобразование из BGR в YUV
    yuv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2YUV)
    show_image(yuv, "yuv")
    # Преобразование из BGR в CMYK
    cmyk = BGR2CMYK(raw_image)
    show_image(cmyk, "cmyk")

    # Разделение на каналы и вывод
    l, a, b = cv2.split(raw_image)
    show_image(l, "l")
    show_image(a, "a")
    show_image(b, "b")

    # Тестирование базовых математических операций над каналом
    l_plus, l_mult, l_minus, l_div = math_with_chanel(l)
    merged_l = cv2.merge([l_div, l_minus, l_mult, l_plus])
    show_image(merged_l, "merged image")

    # Фильтр Blur

    blur = cv2.blur(raw_image, (5, 5))
    show_image(blur, "blur")

    # Фильтр GaussianBlur

    gaussian_blur = cv2.GaussianBlur(raw_image, (5, 5), 0)
    show_image(gaussian_blur, "gaussian_blur")

    # Фильтр Median Blur

    median_blur = cv2.medianBlur(raw_image, 5)
    show_image(median_blur, "median blur")

    # Фильтр BiletteralFilter

    biletteral_filter = cv2.bilateralFilter(raw_image, 9, 75, 75)
    show_image(biletteral_filter, "biletteral filter")

    # Фильтр Canny
    сanny = cv2.Canny(raw_image, 50, 150)
    show_image(сanny, "canny")

    # Фильтр Laplacian
    laplacian = cv2.Laplacian(raw_image, cv2.CV_64F)
    show_image(laplacian, "laplacian")

    # Фильтр Sobel
    sobelx = cv2.Sobel(raw_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(raw_image, cv2.CV_64F, 0, 1, ksize=5)
    combined_sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    show_image(combined_sobel, "combined_sobel")

    # Фильтр Scharr
    scharrx = cv2.Scharr(raw_image, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(raw_image, cv2.CV_64F, 0, 1)
    combined_scharr = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    show_image(combined_scharr, "combined_scharr")


if __name__ == "__main__":
    main()
