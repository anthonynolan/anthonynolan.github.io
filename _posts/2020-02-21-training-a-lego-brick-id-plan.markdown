---
layout: post
title: "Make an image classifier using transfer learning - Part 1"
date: "2020-02-21"
---

In this series of posts I am going to try out the tensorflow `make_image_classifier` command line tool to train a model to recognize pictures of cats and dogs.

Details of how to use the tool are [here](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier)

And I am going to get the pictures of dogs and cats [here](https://www.kaggle.com/c/dogs-vs-cats/data)

As this project is based on transfer learning I probably don't need as many images as are in the kaggle repo - a few hundred of each will suffice.

The make_image_classifier tool is part of tensorflow-hub project. You can find it on pypy [here](https://pypi.org/project/tensorflow-hub/)

To check if you already have the correct version of hub use `pip freeze`

The install usage on the make_image_classifier git page

`pip install "tensorflow-hub[make_image_classifier]~=0.6"`

ensures that you won't get any breaking changes from the .6 version.
