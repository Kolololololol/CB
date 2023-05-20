import numpy as np
import pandas as pd
from support_fucntions import parse_test_mass

# получаю тестовую выборку
df = pd.read_csv("dataset/test.csv")
# перевожу ее в массив
test_mass_csv = np.array(df)

test_mass_without_id = test_mass_csv[:,1:]

print(test_mass_without_id[0])

COLUMN_NAMES, normalize_value_matrix = parse_test_mass(test_mass=test_mass_without_id, df=df)


print(normalize_value_matrix.shape)

df = pd.read_csv("dataset/submission.csv")
df.set