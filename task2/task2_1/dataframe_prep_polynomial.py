import pandas as pd

from datetime import datetime

def dataframe_prep_polynomial(path_to_file):
    '''
    Создание датафрейма

    Используется библиотека pandas, datetime

    '''
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

    return(dataframe)