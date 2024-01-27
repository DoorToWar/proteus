from matplotlib import pyplot  as plt
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

