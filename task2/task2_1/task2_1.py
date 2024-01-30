import polynomial_regression


def main():
    """
    Вызывает функции polynomial_regression

    """

    path_to_file = 'time_messagees.txt'
    polynomial_regression.polynomial_regression(path_to_file)


if __name__ == '__main__':
    main()
