---
layout: post
title: "Make an image classifier using transfer learning - Part 3 (a basic model - tflite)"
date: "2020-02-25"
---

So now that you have downloaded the images of the new classes (cats and dogs in this case) you can run this to generate a new model to classify these.

```
make_image_classifier \
	--image_dir cats_and_dogs_filtered/train \
	--tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 \
	--image_size 224 \
	--saved_model_dir . \
	--labels_output_file class_labels.txt \
	--tflite_output_file new_mobile_model.tflite
```

This gives us a tflite format model and a saved model format one.
For now we will use the saved model format and serve it with tensorflow serving. Details of how to do that are [here](https://www.tensorflow.org/guide/saved_model#running_a_savedmodel_in_tensorflow_serving)
