---
layout: post
title: "Training some classifiers on a public dataset"
date: "2019-09-07 08:30:00 +0100"
---

The [Beast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)) is quite old (1992) but it is an very nice introduction to classification.

The data is all integer values. There are 9 features and a single output - cancer or not.

I did some manual data manipulation first. Grabbed the names file and changed the names to underscores instead of spaces and added commas. I pasted this on top of the data file to allow an easier read by Pandas.

### Train a basic logistic regression

Standard imports first:

```
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV


dat = pd.read_csv('breast_cancer_data/breast-cancer-wisconsin.data')
display(dat.head())
```
![Initial dataframe](/assets/dataframe.png)

The `code` column is just an index type column and is not required.

`dat.drop(['code'], axis=1, inplace=True)`

Let's see if there are any values that we need to replace.

`dat['bare_nuclei'].value_counts()`
![Value counts](/assets/value_counts.png)

So in this column there are 16 instances of '?'. Get rid of them with this `dat = dat[dat['bare_nuclei']!='?']`

You could experiment with replacing them with the mean or some other value, but there are not that many relative to the overall size of the dataset, so missing them is unlikely to impact the final model much.

If you run `.dtypes` on the dataframe you will see that bare_nuclei is an object as opposed to an int like all of the other columns. To convert it to int use this: `dat['bare_nuclei'] = dat['bare_nuclei'].astype('int')`

I want the target column to be 0 for no cancer and 1 for cancer. The values as loaded are 2 and 4 respectively, so use `.map` to change this:

`dat['class'] = dat['class'].map({2:0, 4:1})`

Now put the data into proper variables and split into training and test sets:

```
y = dat['class']
X = dat.drop(['class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Fit a LogisticRegression and calculate the test set score:

```
lr = LogisticRegression(solver='lbfgs', random_state=1)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```

`0.9590643274853801`

Pretty good start.

Let's have a look at what the regression did there:

```
fig = plt.figure(figsize=(10,5))
plt.plot(sorted([a[0] for a in lr.predict_proba(X_test)]), 'r+')
plt.hlines(.5, 0, len(X_test), linestyles='dashed')
plt.show()
```

So this gets the probabability that each point in the test set lies in one or other of the classes. We sort by these probabilities and when we plot we can see a nice logistic function appear.

![Logistic Curve](/assets/logistic_curve.png)

### Next is Principal Component Analysis (PCA)

PCA is a technique for reducing the number of dimensions in your data. It has a few advantages. In many real world data sets most of the variation is contained in a small subset of the features. PCA is not about selecting from the original set of features. Instead it maps the data onto a new set of coordinates which maximally capture the variation. Once you have this new set of coordinates and the transformed data you can decide how many of these components you will use.
PCA can reduce overfitting and can make larger models train much faster.

This code runs PCA for the full range of possible feature numbers and trains a regression on each set. We will look at the scores next.

```
scores = []
for n_comps in range(1,X.shape[1]):
    lr = LogisticRegression(solver='lbfgs', random_state=1)
    p = PCA(n_components=n_comps)
    X_reduced = p.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y)
    lr.fit(X_train, y_train)
    scores.append((lr.score(X_test, y_test), n_comps))    
scores
```

![PCA Scores](/assets/pca_scores.png)

You can see that here just 2 Principal Components gives us the best score. Note that this scoring is raw accuracy and does not use techniques like Cross Validation to give a more stable and repeatable result. It may therefore give a better picture than the underlying reality. However for our purpose of illustrating PCA on a real data set, this is fine.

### Select the best features
In the PCA section above I noted that PCA is not about finding a subset of your existing features, but rather about mapping to a new feature space. The next technique is Select K best and it goes about finding the best subset of your existing features to use.

Select K best works by iterating through every combination of features and training a model. For anything over a handful of features this is computationally infeasible, but the method is a useful aid to our understanding of feature selection.

This code is very similar to the PCA stuff - passing a range of parameter values to the SelectKBest()

```
lr = LogisticRegression(solver='lbfgs')
params = {'skb__k': range(1,X.shape[1])}
skb = SelectKBest()
p = Pipeline([('skb', skb), ('lr', lr)])
g = GridSearchCV(estimator=p, param_grid=params, cv=5)
g.fit(X_train, y_train)
print(g.best_params_)
print(g.score(X_test, y_test))
print(g.best_score_)
```
![Select K Best](/assets/select_k_best.png)

This tells us that we only need 2 features to get 96% accuracy. For very large sets of data with low numbers of features this could be useful in dropping out a lot of unnecessary data that would only slow down training.

This is just a small set of the type of things you can do to build a classifier. Further posts will expand on this to show more advanced methods.
