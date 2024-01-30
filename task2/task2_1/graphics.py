
from matplotlib import pyplot as plt


def graphics(x_scaled, y_scaled, predicted_values):
    """ Построение графиков

    Используется библиотека matplotlib

    Args:
        x_scaled (float): Стандартизированные данные для X
        y_scaled (float): Стандартизированные данные для Y
        predicted_values (float): Предсказанные значения
    """

    # Построение графика
    plt.figure(figsize=(8, 6))  # Увеличение размера графика
    plt.plot(x_scaled, predicted_values, label='Регрессия', color='r')
    # Изменение цвета и стиля линии
    plt.scatter(x_scaled, y_scaled, color='blue', label='Данные', s=5)
    plt.xlabel('Время')
    plt.ylabel('Целевые переменные (стандартизованные)')
    plt.title('График полиномиальной регрессии')
    plt.legend()
    plt.grid()
    plt.show()
