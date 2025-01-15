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

ML Regression in Dash

Dash is the best way to build analytical apps in Python using Plotly figures. 
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

app = Dash(__name__)

models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor}

app.layout = html.Div([
    html.H4("Predicting restaurant's revenue"),
    html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["Regression", "Decision Tree"],
        value='Decision Tree',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input('dropdown', "value"))
def train_and_display(name):
    df = px.data.tips() # replace with your own data source
    X = df.total_bill.values[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        X, df.tip, random_state=42)

    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
    ])
    return fig

app.run_server(debug=True)

Predicting restaurant's revenue -- Regression & Decision Tree algorithm
![image](https://github.com/DoraDeng629/Machine-Learning-Visualization-Python-Plotly-package/blob/main/ML4Regression.png)
![image](https://github.com/DoraDeng629/Machine-Learning-Visualization-Python-Plotly-package/blob/main/ML3decisiontree.png)

