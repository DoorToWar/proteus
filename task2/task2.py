import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def linear_regression(path_to_file = 'time_messagees.txt'):
    '''
    Построение модели линейной регрессии

        Позиционный аргумент - путь к файлу, который хранит кол-во
        сообщений, отправленных в определенное время

        Используются глобальные переменные, необходимые для работы
        другой функции
    '''

    dataframe = dataframe_prep(path_to_file)

    # данные стандартизируются
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(np.array( dataframe['Time']).reshape(-1,1))
    y_scaled = scaler.fit_transform(np.array(dataframe['Amount of messages']).reshape(-1, 1))

    # создаются тренировочные и валидационные выборки
    X_train, X_valid, y_train, y_valid = train_test_split(x_scaled,y_scaled,test_size = 0.25, random_state = 0)

    predicted_values = model_stud(X_train,X_valid,y_train)

    graphics(X_valid,x_scaled,y_scaled,predicted_values)

    # вывод mse 
    print("mean_squared_error is ",mean_squared_error(y_valid, predicted_values))

def graphics(X_valid,x_scaled,y_scaled,predicted_values):
    '''
    Построение графиков

    Используется библиотека matplotlib
    '''

    # Построение графика
    plt.plot(X_valid, predicted_values, label='Модель линейной регрессии', color = 'r')
    plt.scatter(x_scaled, y_scaled, color='b', label='Данные', s = 1)
    plt.xlabel('Время') # подписывается ось x
    plt.ylabel('Целевые переменные (стандартизованные)') # подписывается ось y
    plt.legend()
    plt.grid()
    plt.show()

def dataframe_prep(path_to_file):
    '''
    Создание датафрейма

    Используется библиотека pandas, datetime

    '''
    # Считывается файл
    dataframe = pd.read_csv(path_to_file, delimiter=',', header = None, names = ['Time','Amount of messages'])

    # Данные в столбце Time преобразуются в формат timedelta для последующего вычисления общего количества секунд
    #для обучения модели
    dataframe['Time'] = pd.to_timedelta(dataframe['Time'])
    dataframe['Time'] = dataframe['Time'].dt.total_seconds()

    #Выводится датафрейм
    dataframe.head(80000)

    # Проверяется, есть ли строки  с нулевыми данными
    dataframe.info()
    #Т.к. в столбце Amount of messages хранится количество сообщений, что не может быть не целочисленными значением,
    #эти данные приводятся к типу int
    dataframe['Amount of messages'] = dataframe['Amount of messages'].astype(int)
    # то же самое действие производится с столбцом Time
    dataframe['Time'] = dataframe['Time'].astype(int)

    dataframe.info() # Проверка смены типа данных

    return(dataframe)

def model_stud(X_train,X_valid,y_train):
    '''
    Создание и обучение модели линейной регрессии

    Используется библиотека sklearn
    '''
    # Создаётся модель линейной регрессии
    model_linear_regression = LinearRegression()

    # обучается модель
    model_linear_regression = LinearRegression()
    model_linear_regression.fit(X_train, y_train)

    # предсказываются значения
    predicted_values = model_linear_regression.predict(X_valid)
    return (predicted_values)

def main():
    path_to_file = 'time_messagees.txt'
    linear_regression(path_to_file)

if __name__ == '__main__':
    main()


