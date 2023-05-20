import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

COLUMN_NAMES = {} # словарь header'ов таблицы train выборки
tokenizer_functional_group = None # Токенайзер, будет хранить индексы функциональных групп
normalize_train_matrix = None # Нормализованная train матрица

# Получение header'a выборки
def get_column_header_names(df : pd.DataFrame):
    column_names = list(df.columns.values)
    return [y.strip() for y in column_names]

# Преобразование тренировочного массива в нормализованный вид
def parse_train_mass(train_mass : np.ndarray, df : pd.DataFrame):
    global COLUMN_NAMES
    global normalize_train_matrix
    COLUMN_NAMES = {y: index for index, y in enumerate(get_column_header_names(df))}
    get_mass_of_functional_group(train_mass=train_mass)
    add_to_mass_other_value(train_mass=train_mass)
    return COLUMN_NAMES, normalize_train_matrix

def get_mass_of_functional_group(train_mass : np.ndarray):
    # global COLUMN_NAMES
    global tokenizer_functional_group
    global normalize_train_matrix
    set_of_fg = list(train_mass[:, COLUMN_NAMES.get('Функциональная группа')])
    number_of_fucntional_groups = len(set(set_of_fg))

    tokenizer_functional_group = Tokenizer(num_words=number_of_fucntional_groups, char_level=False)
    tokenizer_functional_group.fit_on_texts(set_of_fg)
    tokenizer_matrix_data = tokenizer_functional_group.texts_to_matrix(set_of_fg)



    normalize_train_matrix = np.zeros((len(tokenizer_matrix_data), number_of_fucntional_groups + 4), dtype=float)
    for i in range(len(tokenizer_matrix_data)):
        for j in range(number_of_fucntional_groups):
            normalize_train_matrix[i, j] = tokenizer_matrix_data[i, j]


def add_to_mass_other_value(train_mass : np.ndarray):
    global normalize_train_matrix
    prioriry_column = COLUMN_NAMES.get("Приоритет")
    status_column = COLUMN_NAMES.get("Статус")
    type_column = COLUMN_NAMES.get("Тип обращения на момент подачи")
    influence_column = COLUMN_NAMES.get("Влияние")
    for i in range(len(normalize_train_matrix)):
        # Нормализация приоритета
        normalize_train_matrix[i, -4] = float(train_mass[i, prioriry_column][0]) / 3.0
        # Нормализация статуса
        normalize_train_matrix[i, -3] = 1 if normalize_train_matrix[i, status_column] == "Закрыт" else 0
        # Нормализация типа обращения на момент подачи
        normalize_train_matrix[i, -2] = 1 if normalize_train_matrix[i, type_column] == "Запрос" else 0
        # Нормализация влияния
        normalize_train_matrix[i, -1] =(float(train_mass[i, influence_column][0]) - 1) / 3.0


def parse_test_answers(train_mass : np.ndarray):
    normalize_test_mass = np.zeros(shape=len(train_mass), dtype=float)
    type_column = COLUMN_NAMES.get("Тип переклассификации")
    for i in range(len(train_mass)):
        normalize_test_mass[i] = float(train_mass[i, type_column])

    return normalize_test_mass