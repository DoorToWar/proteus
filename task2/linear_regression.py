import numpy as np
from dataframe_prep import dataframe_prep
from model_stud import model_stud
from graphics import graphics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def linear_regression(path_to_file = 'time_messagees.txt'):
    '''
    Построение модели линейной регрессии

    Позиционный аргумент - путь к файлу, который хранит кол-во
    сообщений, отправленных в определенное время

    Используется библиотека sklearn
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

