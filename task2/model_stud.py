
from sklearn.linear_model import LinearRegression


def model_stud(X_train, X_valid, y_train):
    """
    Создание и обучение модели линейной регрессии

    Используется библиотека sklearn
    Args:
        X_train (float): Значение Х для обучения модели
        X_valid (float): Значение Х для тестирования модели
        y_train (float): Значение Y для обучения модели

    Returns:
        float: Предсказанные значения
    """
    # Создаётся модель линейной регрессии
    model_linear_regression = LinearRegression()

    # обучается модель
    model_linear_regression = LinearRegression()
    model_linear_regression.fit(X_train, y_train)

    # предсказываются значения
    predicted_values = model_linear_regression.predict(X_valid)
    return (predicted_values)
