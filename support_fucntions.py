import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

COLUMN_NAMES = {} # словарь header'ов таблицы train выборки
tokenizer_functional_group = None # Токенайзер, будет хранить индексы функциональных групп
normalize_train_matrix = None # Нормализованная train матрица
normalize_test_matrix = None # Нормализованная test матрица

# Получение header'a выборки
def get_column_header_names(df : pd.DataFrame):
    column_names = list(df.columns.values)
    return [y.strip() for y in column_names]

# Преобразование тренировочного массива в нормализованный вид
def parse_train_mass(train_mass : np.ndarray, df : pd.DataFrame):
    global COLUMN_NAMES
    global normalize_train_matrix
    COLUMN_NAMES = {y: index for index, y in enumerate(get_column_header_names(df))}
    normalize_train_matrix = np.zeros((len(train_mass),5),dtype=float)
    add_to_mass_other_value(train_mass=train_mass)
    return COLUMN_NAMES, normalize_train_matrix


def parse_test_mass(test_mass : np.ndarray, df : pd.DataFrame):
    global COLUMN_NAMES
    global normalize_test_matrix
    COLUMN_NAMES = {y: index for index, y in enumerate(get_column_header_names(df))}
    normalize_test_matrix = np.zeros((len(test_mass),5),dtype=float)
    add_to_mass_other_value_test(train_mass=test_mass)
    return COLUMN_NAMES, normalize_test_matrix

# def get_mass_of_functional_group(train_mass : np.ndarray):
#     # global COLUMN_NAMES
#     global tokenizer_functional_group
#     global normalize_train_matrix
#     set_of_fg = list(train_mass[:, COLUMN_NAMES.get('Функциональная группа')])
#     number_of_fucntional_groups = len(set(set_of_fg))
#
#     tokenizer_functional_group = Tokenizer(num_words=number_of_fucntional_groups, char_level=False)
#     tokenizer_functional_group.fit_on_texts(set_of_fg)
#     tokenizer_matrix_data = tokenizer_functional_group.texts_to_matrix(set_of_fg)
#
#
#
#     normalize_train_matrix = np.zeros((len(tokenizer_matrix_data), number_of_fucntional_groups + 5), dtype=float)
#     for i in range(len(tokenizer_matrix_data)):
#         for j in range(number_of_fucntional_groups):
#             normalize_train_matrix[i, j] = tokenizer_matrix_data[i, j]


def add_to_mass_other_value(train_mass : np.ndarray):
    print(COLUMN_NAMES)
    global normalize_train_matrix
    prioriry_column = COLUMN_NAMES.get("Приоритет")
    status_column = COLUMN_NAMES.get("Статус")
    type_column = COLUMN_NAMES.get("Тип обращения на момент подачи")
    influence_column = COLUMN_NAMES.get("Влияние")
    critical_column = COLUMN_NAMES.get("Критичность")
    for i in range(len(normalize_train_matrix)):
        # Нормализация критичности
        normalize_train_matrix[i, -5] = (float(train_mass[i, critical_column][0]) - 1) / 3.0
        # Нормализация приоритета
        normalize_train_matrix[i, -4] = float(train_mass[i, prioriry_column][0]) / 3.0
        # Нормализация статуса
        normalize_train_matrix[i, -3] = 1 if train_mass[i, status_column] == "Закрыт" else 0
        # Нормализация типа обращения на момент подачи
        normalize_train_matrix[i, -2] = 1 if train_mass[i, type_column] == "Запрос" else 0
        # Нормализация влияния
        normalize_train_matrix[i, -1] =(float(train_mass[i, influence_column][0]) - 1) / 3.0

def add_to_mass_other_value_test(train_mass : np.ndarray):
    print(COLUMN_NAMES)
    global normalize_test_matrix
    prioriry_column = COLUMN_NAMES.get("Приоритет") - 1
    status_column = COLUMN_NAMES.get("Статус") - 1
    type_column = COLUMN_NAMES.get("Тип обращения на момент подачи") - 1
    influence_column = COLUMN_NAMES.get("Влияние") - 1
    critical_column = COLUMN_NAMES.get("Критичность") - 1
    for i in range(len(normalize_test_matrix)):
        # Нормализация критичности
        normalize_test_matrix[i, -5] = (float(train_mass[i, critical_column][0]) - 1) / 3.0
        # Нормализация приоритета
        normalize_test_matrix[i, -4] = float(train_mass[i, prioriry_column][0]) / 3.0
        # Нормализация статуса
        normalize_test_matrix[i, -3] = 1 if train_mass[i, status_column] == "Закрыт" else 0
        # Нормализация типа обращения на момент подачи
        normalize_test_matrix[i, -2] = 1 if train_mass[i, type_column] == "Запрос" else 0
        # Нормализация влияния
        normalize_test_matrix[i, -1] = (float(train_mass[i, influence_column][0]) - 1) / 3.0


def parse_test_answers(train_mass : np.ndarray):
    normalize_test_mass = np.zeros(shape=len(train_mass), dtype=float)
    type_column = COLUMN_NAMES.get("Тип переклассификации")
    for i in range(len(train_mass)):
        normalize_test_mass[i] = float(float(train_mass[i, type_column]) + 1) / 3

    return normalize_test_mass

