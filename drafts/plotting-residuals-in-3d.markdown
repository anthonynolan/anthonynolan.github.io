---
layout: post
title: "Plotting residuals in 3D"
date: "2019-09-10"
---
Finding the best fit line for a linear regression is done by minimizing the sum of the squared residuals from the plot. These are the vertical distances between the regression line and each observed point. For regressions with multiple input features a technique like gradient descent can be used to minimize this function. However for a simple regression with one input variable, you can use python to go through all of the possible regression lines - every slope and every intercept - calculate the residuals and square and sum them. Finding the lowest of these residuals solves the problem for you.

Note that this is really an exercise in 3D plotting and an aid to understanding linear regression. You would never solve a real regression this way - even a simple one. Use `sklearn.linear_model.LinearRegression` for example instead.

First all of the imports that we will need

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

Then we need some test data which 'looks right'. Rather than go to the trouble of finding real data we can use a nice helper that sklearn provides: [make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html). make_regression is one of several very handy data generators in `sklearn.datasets` that are worth getting familiar with.

```
X, y = make_regression(n_samples=50,n_features=1, noise=25)
```

This is straightforward enough. If you leave the noise parameter out you get a dead straight regression. Nice in the real world, but not useful here.

```
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```

To see your intercept and slope use this:

```
lr.coef_,lr.intercept_
```
You will need these for later to see that your 3d graph is correct.

Next plot the regression line, the observed points and the residuals.

```
X_for_pred = np.linspace(X_train.min(), X_train.max(), 250).reshape(-1,1)
y_pred = lr.predict(X_for_pred)
fig = plt.figure(figsize=(12,8))
plt.scatter( X_train,y_train, color='b', s=150, marker='x', label='observations')
plt.xlabel('Predictor')
plt.ylabel('Response')
plt.scatter(X_for_pred, y_pred, color='r', label='best fit', s=1)
plt.legend()

for index in range(len(X_train)):
    plt.vlines(X_train[index], y_train[index], lr.predict(X_train[index].reshape(-1,1)), colors='g')
plt.show()
```
![Residuals plot](/assets/residuals-plot.png)

So that is the nice regression using sklearn. Next we will create a range of values for the slope (m) and intercept of our line (c). To choose these I fiddled around to make sure my minimum residual was included. Also defined here is x. This is just a range of values that we will evaluate for each set a parameters. So just the domain of the function.

```
params = []
for c_val in np.linspace(-40,40,20):
    for m_val in np.linspace(0,150,20):
        params.append((m_val, c_val))
```

Iterate through every combination of c and m and calculate the residual at each point in x.

```
residuals = []
for m, c in params:
    residuals_summed = 0
    for x, y in zip(X_train, y_train):
        residual_at_point = y - (m*x + c)
        residuals_summed += residual_at_point**2
    residuals.append((m, c, residuals_summed))
```

We can plot the residuals at this point. They are just in a list, so we have lost the 2 axes c and m. But the plot is worth doing as it shows us that we are going in the right direction.

```
plt.plot([a[2] for a in residuals])
```

![Unravelled residuals](/assets/unravelled-residuals.png)

Because it is unravelled there are lots of local minima. However you can see that somewhere in the middle is the one we want. The `residuals` list still contains out c and m values, so we can use them in a better plot.

```
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(m_domain, c_domain)
Z = np.array([a[2] for a in residuals]).reshape(20,20)
ax.plot_surface(X, Y, Z, alpha=.5)
ax.set_xlabel('Slope')
ax.set_ylabel('intercept')
ax.set_zlabel('RSS')
ax.plot([np.min(X),np.max(X)], [lr.intercept_]*2, zs=0, zdir='z', label='Minimun RSS', color='g')
ax.plot( [lr.coef_]*2,[np.min(Y),np.max(Y)], zs=0, zdir='z', color='g')
ax.legend()
plt.show()
```
![3d residual plot](/assets/3d-residual-plot.png)

You can see a clear minimum here from which you can read off the values of slope and intercept which give the best fit. 
