import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

global x_scaled, y_scaled, X_valid, predicted_values

def polynomial_regression(path_to_file='time_messagees.txt'):
    
    '''
    Построение модели полиномиальной регрессии

    Позиционный аргумент - путь к файлу, который хранит кол-во
    сообщений, отправленных в определенное время

    Используются глобальные переменные, необходимые для работы
    другой функции
    '''
    global x_scaled, y_scaled, X_valid, predicted_values

    # Считывается файл
    dataframe = pd.read_csv(path_to_file, delimiter=',', header=None, names=['Time', 'Amount of messages'])

    # Данные в столбце Time преобразуются в формат timedelta для последующего вычисления общего количества секунд
    # для обучения модели
    dataframe['Time'] = pd.to_timedelta(dataframe['Time'])
    dataframe['Time'] = dataframe['Time'].dt.total_seconds()

    # Выводится датафрейм
    print(dataframe.head(80000))

    # Проверяется, есть ли строки с нулевыми данными
    dataframe.info()

    # Т.к. в столбце Amount of messages хранится количество сообщений, что не может быть не целочисленными значением,
    # эти данные приводятся к типу int
    dataframe['Amount of messages'] = dataframe['Amount of messages'].astype(int)

    # то же самое действие производится с столбцом Time
    dataframe['Time'] = dataframe['Time'].astype(int)

    dataframe.info() # Проверка смены типа данных

    # Создаётся модель полиномиальной регрессии
    model_polynomial_regression = PolynomialFeatures(degree=2)

    # данные стандартизируются
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(np.array(dataframe['Time']).reshape(-1, 1))
    y_scaled = scaler.fit_transform(np.array(dataframe['Amount of messages']).reshape(-1, 1))

    # преобразуем признаки в полиномиальные
    x_polynomial = model_polynomial_regression.fit_transform(x_scaled)

    # создаются тренировочные и валидационные выборки
    X_train, X_valid, y_train, y_valid = train_test_split(x_polynomial, y_scaled, test_size=0.25, random_state=0)

    # обучается модель
    model_polynomial_regression = LinearRegression()
    model_polynomial_regression.fit(X_train, y_train)

    # предсказываются значения
    predicted_values = model_polynomial_regression.predict(x_polynomial)

    # вывод mse
    print("mean_squared_error is ", mean_squared_error(y_scaled, predicted_values))

def graphics():
    '''
    Построение графиков
    Используется библиотека matplotlib
    '''

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

def main():
    '''
    Вызывает функции polynomial_regression и graphics
    '''
    polynomial_regression()
    graphics()

main()
