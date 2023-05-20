import keras
import keras.backend as K
from keras.losses import mean_squared_error
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from support_fucntions import parse_train_mass, parse_test_answers

global RNN_model

# получаю тренировочную выборку
df = pd.read_csv("dataset/train.csv")
# перевожу ее в массив
train_mass_csv = np.array(df)
train_mass_csv_without_first_row = train_mass_csv[1:]

cleaned_train_mass_csv = [train_mass_csv[0]]

for row in train_mass_csv_without_first_row:
    if (row[0] != cleaned_train_mass_csv[-1][0]):
        cleaned_train_mass_csv.append(row)

cleaned_train_mass_csv_numpy = np.array(cleaned_train_mass_csv)


train_mass_equal = []

count_0, count_1, count_2 = 0, 0, 0

for row in cleaned_train_mass_csv_numpy:
    match row[10]:
        case 0:
            if (count_0 < 200):
                train_mass_equal.append(row)
                count_0 += 1
        case 1:
            if (count_1 < 200):
                train_mass_equal.append(row)
                count_1 += 1
        case 2:
            if (count_2 < 200):
                train_mass_equal.append(row)
                count_2 += 1

train_mass_equal = np.array(train_mass_equal) # получил уравновешенную выборку переклассификаций


# Парсим и нормируем train матрицу
COLUMN_NAMES, normalize_train_matrix = parse_train_mass(train_mass=train_mass_equal, df=df)

normalize_test_matrix = parse_test_answers(train_mass=train_mass_equal)


# Кастомная f1 scope метрика
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
#
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def getModel():
    model = keras.Sequential([
        tf.keras.layers.Input(shape=len(normalize_train_matrix[0])),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
#
    model.compile(optimizer=tf.optimizers.Ftrl, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy", f1_metric])
#
    return model
#
#
# RNN_model = getModel()
#
# RNN_model.fit(normalize_train_matrix, normalize_test_matrix, batch_size=32, epochs=100)



# number_of_train_elements = math.floor(train_mass.shape[0] * 0.75)
#
# # train.csv поделенный на обучающую и тестовую выборки
# true_train_mass = train_mass[0:number_of_train_elements, :]
# true_test_mass = train_mass[number_of_train_elements:, :]
#
# print(f"true_train_mass.shape = {true_train_mass.shape}")
# print(f"true_test_mass.shape = {true_test_mass.shape}")
#
#
# df = pd.read_csv("dataset/test.csv")
# test_mass = np.array(df)
#
# # приведение test к формату train
# test_mass_without_id = test_mass[:,1:] #???
#
# df = pd.read_csv("dataset/submission.csv")
# submission_mass = np.array(df)
#
# print(f"train_mass.shape = {train_mass.shape}")
# print(f"test_mass.shape = {test_mass.shape}")
# print(f"test_mass_without_id = {test_mass_without_id.shape}")
# print(f"submission.shape = {submission_mass.shape}")
#
