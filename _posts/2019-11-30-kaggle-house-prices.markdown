---
layout: post
title: "House Price Predictions for Kaggle"
date: "2019-12-01"
---
Link to [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)


```python
import pandas as pd
```

First join the train and test data together. This means we can encode the categorical variables and not have to worry about different categories appearing in our test data.
We also fill in any missing data with means here.


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat([train, test], axis=0, sort=False)
all_data = all_data.fillna(all_data.mean())
all_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



Quick way to get all numeric columns (we call them `cont_vars` here for continuous variables), default behaviour of [describe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) is only to include numeric columns

note `;` and var name after line to echo value in notebook neatly


```python
cont_vars = all_data.describe().columns.tolist();
cont_vars[:5]
```




    ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual']



Same approach for the categorical variables here. We are assuming that in this data all columns of dtype `object` are categorical. If there were freetext columns like comments they would need to be omitted from this list and possibly dealt with in a different way.


```python
cat_vars = all_data.describe(include='object').columns.tolist(); cat_vars[:5]
```




    ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour']



Get the one hot encoding - dummy variables - for all categorical columns at once


```python
all_data = pd.concat([pd.get_dummies(all_data[cat_vars]), all_data[cont_vars]], axis=1)
```

We won't use train_test_split here as we want to pull out the original test set. Note that y_test is missing from this list as it does not make sense. We had Nan values in the target column and we filled them with .mean earlier, so that is junk data.


```python
#Split by row position now
train = all_data[:len(train)]
test = all_data[len(train):]

X_train = train.drop('SalePrice', axis=1)
X_test = test.drop('SalePrice', axis=1)

y_train = train['SalePrice']
```

Now we do use train_test_split, but this is just so that we can get a suitable y_test to score our model on. Variable naming a bit clunky here, but these are just throwaways.


```python
X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(X_train, y_train)
```

`LinearRegression` first. I am usually surprised at how well this works. Same here.


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_local, y_train_local)
model.score(X_test_local, y_test_local)
```




    0.8754440101764568



Next Lasso. Lasso uses regularization to reduce overfitting. The parameter alpha is used to control the degree of this regularization. Here we will iterate over a linear space of alphas to find the best one. A GridSearchCV would probably be better, but this suits our needs for now.


```python
from sklearn.linear_model import Lasso
alphas = np.linspace(1,1000, 50)
scores = []
for alpha in np.linspace(1,1000, 50):
    model = Lasso(alpha=alpha, max_iter=1000, tol=.1)
    model.fit(X_train_local, y_train_local)
    score = model.score(X_test_local, y_test_local)
    scores.append(score)
```


```python
plt.scatter( alphas, scores)
plt.grid(axis='x')
```


![png](/assets/output_18_0.png)


Reading through the results it looks like an alpha of 327 gives the best results. We will retrain with the full dataset (not the _local ones). No scoring here as we don't have an effective y_test.


```python
model = Lasso(alpha=327, max_iter=10000)
model.fit(X_train, y_train)
```




    Lasso(alpha=327, copy_X=True, fit_intercept=True, max_iter=10000,
          normalize=False, positive=False, precompute=False, random_state=None,
          selection='cyclic', tol=0.0001, warm_start=False)



Finally create the submission for Kaggle


```python
pd.DataFrame({'id':test.Id, 'SalePrice':model.predict(X_test)}).to_csv('output.csv', index=False)
```


```python

```
