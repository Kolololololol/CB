import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ВТОРЫЕ ГРАФИКИ

COL_PRIORITY = 2
COL_NEW_CLASSIFICATION = 10
COL_CRITICAL = 13
COL_INFLUENCE = 14

df = pd.read_csv("dataset/train.csv")

train_mass_csv = np.array(df)

new_classification_array = []
priority_array = []
influence_array = []
critical_array = []

for row in train_mass_csv:
    new_classification_array.append(int(row[COL_NEW_CLASSIFICATION]))
    priority_array.append(int(row[COL_PRIORITY][0]))
    influence_array.append(int(row[COL_INFLUENCE][0]))
    critical_array.append(int(row[COL_CRITICAL][0]))


fig, ax = plt.subplots(nrows=3, ncols=1)

ax[0].set_xlabel("Переклассификация")
ax[0].set_ylabel("Приоритет")

ax[1].set_xlabel("Переклассификация")
ax[1].set_ylabel("Влияние")

ax[2].set_xlabel("Переклассификация")
ax[2].set_ylabel("Критичность")


ax[0].scatter(new_classification_array, priority_array)
ax[1].scatter(new_classification_array, influence_array)
ax[2].scatter(new_classification_array, critical_array)



plt.show()