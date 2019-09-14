---
layout: post
title: "Basic neural network for classification"
date: "2019-09-14"
---

This is a fisheries dataset from Tasmania. Working out how old a particular abalone is is a time consuming task. The goal is to determine what measures could act as a proxy for age and so replace the time consuming work.
The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Abalone).

A few imports and a quick look at the data.

```
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dat = pd.read_csv('abalone_data/abalone.data')
print(len(dat))
display(dat.head())
```
![Abalone dataset](/assets/abalone-dataset.png)

Sex is a categorical variable given as M, F and I. So convert them to integers first.

```
dat.Sex = dat['Sex'].map({'M':0, 'F':1, 'I':2})
```
Map the ring sizes to categories - basically small, medium and large. This gives much better results then trying to train a model with individual ring count as a target.
Then turn those ring ints into one hot vectors using to_categorical in keras.utils.
Finally split the dataset up into test and train.

```
X = dat.drop('Rings', axis=1)
y = dat['Rings']
def ring_int_mapper(ring_count):
    if ring_count<9:
        return 0
    if ring_count< 11:
        return 1
    else:
        return 2
y_cat = to_categorical(y.map(ring_int_mapper))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat)
```

This is the neural network code.
```
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
s = Sequential()
s.add(Dense(100, input_dim=X.shape[1], activation='relu'))
s.add(Dropout(.5))
s.add(Dense(70, activation='relu'))
s.add(Dropout(.5))
s.add(Dense(10, activation='relu'))
s.add(Dense(3, activation='softmax'))
s.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = s.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
```
It is pretty easy to read.
Once it is trained this code will plot the loss and accuracy. They go in the right direction.

```
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.show()
print('final accuracy {}'.format(history.history['acc'][-1]))
```
![Abalone accuracy](/assets/abalone-accuracy.png)

The result for a logistic regression on this dataset with 3 classes of ring size is
0.6555023923444976
So no benefit from using this basic neural network.
That said this is completely untuned. Next I would probably examine some of the incorrectly predicted test values and see if I can see some spot of pattern.
