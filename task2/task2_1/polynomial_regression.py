import dataframe_prep_polynomial
import numpy as np
import graphics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression(path_to_file):
    """Построение модели полиномиальной регрессии

    Args:
        path_to_file (str): Путь к файлу
    """

    dataframe = dataframe_prep_polynomial.dataframe_prep_polynomial(
        path_to_file)

    # Создаётся модель полиномиальной регрессии
    model_polynomial_regression = PolynomialFeatures(degree=4)

    # данные стандартизируются
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(np.array(dataframe['Time']).reshape(-1, 1))
    y_scaled = scaler.fit_transform(
        np.array(dataframe['Amount of messages']).reshape(-1, 1))

    # преобразуем признаки в полиномиальные
    x_polynomial = model_polynomial_regression.fit_transform(x_scaled)

    # создаются тренировочные и валидационные выборки
    X_train, X_valid, y_train, y_valid = train_test_split(
        x_polynomial, y_scaled, test_size=0.25, random_state=0)

    # обучается модель
    model_polynomial_regression = LinearRegression()
    model_polynomial_regression.fit(X_train, y_train)

    # предсказываются значения
    predicted_values = model_polynomial_regression.predict(x_polynomial)

    graphics.graphics(x_scaled, y_scaled, predicted_values)

    # вывод mse
    print("mean_squared_error is ", mean_squared_error(
        y_scaled, predicted_values))
