---
layout: post
title: "Titanic Survival Predictions for Kaggle"
date: "2019-12-07"
---
This is a solution to the [Kaggle Titanic challenge](https://www.kaggle.com/c/titanic).

First imports and join the training and test data together for preprocessing.


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat([train, test], axis=0, sort=False)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.0</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Remove the unnecessary columns and change passender class to categorical.


```python
all_data.fillna(all_data.mean(), inplace=True)
all_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
all_data['Pclass'] = all_data['Pclass'].astype('category')
```


```python
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>



Carry out our one hot encoding and split the data back out into training and test sets for X and y.


```python
all_data = pd.get_dummies(all_data)
X = all_data[:len(train)].drop('Survived', axis=1)
y = all_data[:len(train)]['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## Logistic Regression

First model to try. Gives 78% accuracy on test set.

Also using the .cv_results_ attribute of the trained model to get a dataframe of the cross validation results. Veru handy for plotting and seeing where you are in a readable format.


```python
pl = Pipeline([('lr', LogisticRegression(solver='liblinear', random_state=42))])
params={'lr__penalty':['l1','l2'], 'lr__max_iter': [300], 'lr__C':[.01,.05,1]}
model = GridSearchCV(pl, param_grid = params, cv=5, verbose=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.best_estimator_)
results = pd.DataFrame(model.cv_results_)
results
```

    0.7892376681614349
    Pipeline(memory=None,
             steps=[('lr',
                     LogisticRegression(C=1, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=300,
                                        multi_class='warn', n_jobs=None,
                                        penalty='l1', random_state=42,
                                        solver='liblinear', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)





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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_lr__C</th>
      <th>param_lr__max_iter</th>
      <th>param_lr__penalty</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003587</td>
      <td>0.000538</td>
      <td>0.001305</td>
      <td>0.000152</td>
      <td>0.01</td>
      <td>300</td>
      <td>l1</td>
      <td>{'lr__C': 0.01, 'lr__max_iter': 300, 'lr__pena...</td>
      <td>0.649254</td>
      <td>0.671642</td>
      <td>0.656716</td>
      <td>0.624060</td>
      <td>0.676692</td>
      <td>0.655689</td>
      <td>0.018617</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.003520</td>
      <td>0.000474</td>
      <td>0.001136</td>
      <td>0.000056</td>
      <td>0.01</td>
      <td>300</td>
      <td>l2</td>
      <td>{'lr__C': 0.01, 'lr__max_iter': 300, 'lr__pena...</td>
      <td>0.701493</td>
      <td>0.776119</td>
      <td>0.768657</td>
      <td>0.736842</td>
      <td>0.751880</td>
      <td>0.747006</td>
      <td>0.026550</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003547</td>
      <td>0.000273</td>
      <td>0.001185</td>
      <td>0.000085</td>
      <td>0.05</td>
      <td>300</td>
      <td>l1</td>
      <td>{'lr__C': 0.05, 'lr__max_iter': 300, 'lr__pena...</td>
      <td>0.768657</td>
      <td>0.820896</td>
      <td>0.776119</td>
      <td>0.781955</td>
      <td>0.827068</td>
      <td>0.794910</td>
      <td>0.024163</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.003367</td>
      <td>0.000227</td>
      <td>0.001114</td>
      <td>0.000017</td>
      <td>0.05</td>
      <td>300</td>
      <td>l2</td>
      <td>{'lr__C': 0.05, 'lr__max_iter': 300, 'lr__pena...</td>
      <td>0.791045</td>
      <td>0.813433</td>
      <td>0.783582</td>
      <td>0.804511</td>
      <td>0.834586</td>
      <td>0.805389</td>
      <td>0.017875</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.003812</td>
      <td>0.000154</td>
      <td>0.001172</td>
      <td>0.000126</td>
      <td>1</td>
      <td>300</td>
      <td>l1</td>
      <td>{'lr__C': 1, 'lr__max_iter': 300, 'lr__penalty...</td>
      <td>0.791045</td>
      <td>0.820896</td>
      <td>0.783582</td>
      <td>0.804511</td>
      <td>0.842105</td>
      <td>0.808383</td>
      <td>0.021083</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.003586</td>
      <td>0.000313</td>
      <td>0.001156</td>
      <td>0.000087</td>
      <td>1</td>
      <td>300</td>
      <td>l2</td>
      <td>{'lr__C': 1, 'lr__max_iter': 300, 'lr__penalty...</td>
      <td>0.791045</td>
      <td>0.813433</td>
      <td>0.783582</td>
      <td>0.804511</td>
      <td>0.834586</td>
      <td>0.805389</td>
      <td>0.017875</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Lasso performs poorly.


```python
from sklearn.linear_model import Lasso
pl = Pipeline([('lasso', Lasso())])
params = {'lasso__alpha': np.linspace(.01, 100, 10)}
model = GridSearchCV(estimator=pl, param_grid=params, cv=5, iid=False)
model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    0.39011591747098123



Random Forest is better than the Logistic.


```python
pl = Pipeline([('pca', PCA()), ('rf', RandomForestClassifier(max_features=None, max_depth=None, min_samples_split=4, n_estimators=100))])

params = {'rf__n_estimators': [50], 'pca__n_components': [4, 5]}
params = {}
grid = GridSearchCV(pl, param_grid=params, cv=5)
grid.fit(X_train, y_train)
print('score {}'.format(grid.score(X_test, y_test)))
```

    score 0.7982062780269058


SKLearn's build in Gradient boosting classier does ok. Nothing special.


```python
pl = Pipeline([('gbc',GradientBoostingClassifier(n_estimators=500, random_state=1, max_depth=1) )])
params = {'gbc__n_estimators':[50,100,5000], 'gbc__learning_rate': [1,.1,.01]}
grid = GridSearchCV(estimator=pl, param_grid=params, cv=3, verbose=0)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)
```




    0.7892376681614349



A tuned XGBoost gives our best result


```python
from xgboost import XGBClassifier
pl = Pipeline([('gbc', XGBClassifier())])
params = {'gbc__max_depth':[2,3,4, 5], 'gbc__n_estimators':[200,300,400, 500], 'gbc__learning_rate':[.05, .01, .001]}
model = GridSearchCV(estimator=pl, param_grid = params, cv=3, verbose=0)
model.fit(X_train, y_train)
print(model.best_estimator_)
model.score(X_test, y_test)
```

    Pipeline(memory=None,
             steps=[('gbc',
                     XGBClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=1, gamma=0, learning_rate=0.001,
                                   max_delta_step=0, max_depth=4,
                                   min_child_weight=1, missing=None,
                                   n_estimators=400, n_jobs=1, nthread=None,
                                   objective='binary:logistic', random_state=0,
                                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                   seed=None, silent=None, subsample=1,
                                   verbosity=1))],
             verbose=False)





    0.8251121076233184



For completeness we will do a simple Naive Bayes and a DecisionTreeClassifier. Both of these underperform the XGB.


```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_test, y_test)
```




    0.7847533632286996




```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt.score(X_test, y_test)
```




    0.7892376681614349



Make predictions on a model and produce the output file in the right format for submission to Kaggle.


```python
predictions = model.predict(all_data[len(train):].drop('Survived', axis=1))
with open('output.csv', 'w') as f:
    f.write('PassengerId,Survived\n')
    for passender_id, prediciton in zip(test['PassengerId'], predictions):
        f.write('{},{}\n'.format(passender_id, int(prediciton)))
```
