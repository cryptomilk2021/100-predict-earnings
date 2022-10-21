import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:,.2f}'.format
df_data = pd.read_csv('NLSY97_subset.csv')

print(df_data.shape)
df_data = df_data.drop_duplicates(subset=['ID'])

nbr_rows_to_keep = int(len(df_data) / 5)
df = df_data.head(nbr_rows_to_keep)
df.to_csv('small_sample.csv')


