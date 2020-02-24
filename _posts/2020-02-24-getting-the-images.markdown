---
layout: post
title: "Make an image classifier using transfer learning - Part 2 (getting the images)"
date: "2020-02-24"
---

Few imports and then load the data from a google folder - extract also.

```python
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

_URL = 'http://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
```
    Downloading data from http://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
    68608000/68606236 [==============================] - 48s 1us/step

```python
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
PATH
```
    '/Users/anthony.nolan/.keras/datasets/cats_and_dogs_filtered'

You can see the first 5 file names in the dogs directory
```python
os.listdir(os.path.join(PATH,'train', 'dogs'))[:5]
```

    ['dog.775.jpg', 'dog.761.jpg', 'dog.991.jpg', 'dog.749.jpg', 'dog.985.jpg']

And the shape of a specific image using PIL and numpy
```python
pic = Image.open('/Users/anthony.nolan/.keras/datasets/cats_and_dogs_filtered/train/dogs/dog.775.jpg')
np.array(pic).shape
```




    (500, 448, 3)



Here is the raw numpy array data
```python
np.array(pic)
```




    array([[[ 97,  57,  47],
            [ 96,  58,  47],
            [ 97,  59,  48],
            ...,
            [ 69,  76,  82],
            [ 69,  76,  82],
            [ 69,  76,  82]],

           [[ 97,  57,  45],
            [ 97,  59,  48],
            [ 97,  59,  48],
            ...,
            [ 69,  76,  82],
            [ 69,  76,  82],
            [ 69,  76,  82]],

           [[ 99,  59,  47],
            [ 99,  59,  47],
            [ 98,  60,  49],
            ...,
            [ 68,  77,  82],
            [ 68,  77,  82],
            [ 68,  77,  82]],

           ...,

           [[155, 144, 140],
            [152, 141, 137],
            [148, 137, 131],
            ...,
            [171, 166, 163],
            [170, 165, 162],
            [170, 165, 162]],

           [[153, 142, 138],
            [150, 139, 135],
            [146, 135, 129],
            ...,
            [171, 166, 163],
            [170, 165, 162],
            [170, 165, 162]],

           [[151, 140, 136],
            [149, 138, 134],
            [144, 133, 127],
            ...,
            [171, 166, 163],
            [170, 165, 162],
            [170, 165, 162]]], dtype=uint8)




```python
pic
```



And finally the image
![dog](/assets/output_5_0.png)




```python

```
