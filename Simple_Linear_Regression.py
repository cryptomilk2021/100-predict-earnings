import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:,.2f}'.format
# df_data = pd.read_csv('NLSY97_subset.csv')
df_data = pd.read_csv('small_sample.csv')

# S = years of schooling
# EARNINGS = Current hourly earnings in $
df = pd.DataFrame()
df['Years of schooling'] = df_data['S'].copy()
df['Hourly earnings'] = df_data['EARNINGS'].copy()

y = df['Hourly earnings'].values.reshape(-1, 1)
X = df['Years of schooling'].values.reshape(-1, 1)

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)


def calc(slope, intercept, hours):
    return slope * hours + intercept


print(
    f'after 1 year more of study a person is expected to earn ${round(float(calc(regressor.coef_, regressor.intercept_, 10) - calc(regressor.coef_, regressor.intercept_, 9)), 2)} more per hour')

y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
exit()

df.plot.scatter(x='Years of schooling', y='Hourly earnings', title='Years of schooling vs Hourly rates')
plt.show()
