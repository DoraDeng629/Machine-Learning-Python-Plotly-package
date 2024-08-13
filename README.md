This project shows how to use Plotly charts for displaying various types of regression models, starting from simple models like Linear Regression, and progressively move towards models like Decision Tree and Polynomial Features. We highlight various capabilities of plotly, such as comparative analysis of the same model with different parameters, displaying Latex, surface plots for 3D data, and enhanced prediction error analysis with Plotly Express.

I will use Scikit-learn to split and preprocess our data and train various regression models. Scikit-learn is a popular Machine Learning (ML) library that offers various tools for creating and training ML algorithms, feature engineering, data cleaning, and evaluating and testing models. It was designed to be accessible, and to work seamlessly with popular libraries like NumPy and Pandas.

Ordinary Least Square (OLS) with plotly.express
This example shows how to use plotly.express's trendline parameter to train a simply Ordinary Least Square (OLS) for predicting the tips waiters will receive based on the value of the total bill.

import plotly.express as px

df = px.data.tips()
fig = px.scatter(
    df, x='total_bill', y='tip', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
)
fig.show()
![image](https://github.com/DoraDeng629/Machine-Learning-Visualization-Python-Plotly-package/blob/main/ML1.png)


Linear Regression with scikit-learn
You can also perform the same prediction using scikit-learn's LinearRegression.

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.tip)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()
![image](https://github.com/DoraDeng629/Machine-Learning-Visualization-Python-Plotly-package/blob/main/ML2regressionfit.png)
