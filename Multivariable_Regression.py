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
# EXP = years of work experience

df = pd.DataFrame()
df['Hourly earnings'] = df_data['EARNINGS'].copy()
df['Years of schooling'] = df_data['S'].copy()
df['Years of work experience'] = df_data['EXP'].copy()

y = df['Hourly earnings']
X = df[['Years of schooling', 'Years of work experience']]

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

feature_names = X.columns

model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients,
                              index = feature_names,
                              columns = ['Coefficient value'])

edu_coefficient = coefficients_df.iloc[0].tolist()

work_coefficient = coefficients_df.iloc[1].tolist()

print(f'after 16 years of study and 5 years of work, we predict to earn ${round(5 * work_coefficient[0] + 16 * edu_coefficient[0], 2)} per hour')

exit()

y_pred = regressor.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
results.to_csv('predicted.csv')
exit()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)
print(f'regressor.score(X_test, y_test) = {regressor.score(X_test, y_test)}')
exit()



def calc(slope, intercept, hours):
    return slope * hours + intercept


# print(
#     f'after 1 year more of study a person is expected to earn ${round(float(calc(regressor.coef_, regressor.intercept_, 10), 2)} more per hour')

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
