---
layout: post
title: "Training word embeddings on the novels of Charles Dickens"
date: "2020-09-15"
---

We are going to try to train a set of word embeddings using the novels of Charles Dickens as training data. We will start with a single novel - Bleak House. Once we have a basic working pipeline we can expand this to a larger set of novels.

### Some basic cleaning of the data first
This function strips our single quotes replacing them with blank. It also strips out a larger set of punctuation characters and replaces those with space.
The reason to handle these 2 types of character differently is that replacing single quote with space breaks up words like `don't` into `don` and `t`. If the second class of character is replaced with blank, then some contiguous words are shoved together giving an nonsense word which our model won't be able to do anything with.


```
def process(st):
    strip_chars = ['\'']
    for c in strip_chars:
        st = st.replace(c , '')

    rep_with_space_chars = [',', '.', '!', '?', '"', '-', ';', '(', ')']
    for c in rep_with_space_chars:
        st = st.replace(c , ' ')

    word_list = word_tokenize(st)
    word_list = [porterStemmer.stem(a.lower().strip()) for a in word_list]
    return word_list
```

### Filter out 'unknown words'

Some words will appear very infrequently in our corpus. Including them will make our data more sparse, our training time considerably longer and our model has no hope of predicting things like [this](https://en.wikipedia.org/wiki/Hapax_legomenon). We will replace these characters with a token `<UNK>` in our case. We pick a minimum frequency of occurrence first and then filter out any words which appear less then this.

```
def get_most_common(word_array):
    most_common_count = len({k:v for k, v in FreqDist(word_array).items() if v>minimum_frequency})
    c = Counter(words)
    most_common = [pair[0] for pair in c.most_common(most_common_count)]
    return most_common, most_common_count

def replace_unk(word_array, most_common_words):    
    words_with_unk = [word if word in most_common else unknown_word_token for word in word_array]
    return words_with_unk
```

We use the FreqDist class from nltk to find our most common words.


### Convert the body of text to 'examples'

Our model needs to be fed large numbers of examples. We are working on supervised learning here, so they consist of some input features and a target value. We will select sets of 5 words. 2 leading and 2 trailing a centre word. The centre word is the target. So we will rearrange the examples to a,b,c,d,e -> a,b,d,e,c

Once the first 5 words are converted to an example we slide the window along by a single word and repeat the process. Continuing this sliding window for the rest of the text.

```
def convert_window(words_with_unk):
    input = []
    for a in range(context_size, len(words_with_unk)-context_size):
        x = [item for sublist in [words_with_unk[a-context_size:a], words_with_unk[a+1:a+context_size+1], [words_with_unk[a]]] for item in sublist]
        input.append(x)
    return input
```

Our output from this process is of the correct shape, but is still words.

### Convert words to numbers

This method takes the examples array and converts all of the words to their indices in the vocabulary

```
def convert_word_data_to_numbers(input):
    Xs = []
    Ys = []
    for row in input:
        Xs.append([word_to_index[word] for word in row[:-1]])
        Ys.append([word_to_index[word] for word in row[-1:]])

    X = np.vstack(Xs)
    Y = np.vstack(Ys)

    X_train_incl_val, X_test, Y_train_incl_val, Y_test = train_test_split(X, Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_incl_val, Y_train_incl_val)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
```

### Build the model

Now we have everything we need to build our model. We will use the keras Embedding layer as the first layer in our model. The choice of embedding dimension is a hyperparameter that we need to tune manually. Roughly speaking larger dimensions can capture finer detail in the training set, but take more examples to converge. The weights in the embedding are initialised according to this keras initialiser https://keras.io/api/layers/initializers/#randomuniform-class As we train the model they will converge to a set of embedding vectors. One vector for each word in our vocabulary.

```
def build_model():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=len(vocab), output_dim=8))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(len(vocab), activation='softmax', name='output-layer'))

    model.compile(
    optimizer=keras.optimizers.RMSprop(),  
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return model
```

Note the second layer is an averaging layer. Its purpose is to take the average of the embeddings of the 4 input words and use that average for subsequent layers.

### Results of our training
So this is what our out put from keras looks like for training that network on a single novel.

```
Train on 203152 samples, validate on 67718 samples
Epoch 1/2
203152/203152 [==============================] - 115s 567us/sample - loss: 5.7066 - sparse_categorical_accuracy: 0.0855 - val_loss: 5.5489 - val_sparse_categorical_accuracy: 0.1040
Epoch 2/2
203152/203152 [==============================] - 115s 564us/sample - loss: 5.5021 - sparse_categorical_accuracy: 0.1073 - val_loss: 5.5186 - val_sparse_categorical_accuracy: 0.1105
```

Our validation accuracy is about 11%. Pretty poor, but we did not expect much with such a small dataset. We will continue to run some predictions and view the embeddings. In later posts the model will be refined.

### Prediction

To carry out a prediction we pass a sentence into the same preprocessing steps as used to set up the training data.

* Strip unnecessary chars
* Apply porterStemmer
* Replace the unknown words with the unknown token
* Convert the sentence to a window of data

We will choose a 5 word sentence for simplicity here.


```
def predict(sentence, model):
    input = convert_window(replace_unk(process(sentence), most_common))[0]
    indices = [word_to_index[word] for word in input]
    prediction = model.predict(indices[:-1])
    averaged_predictions = layers.GlobalAveragePooling1D()(np.expand_dims(prediction, 0)).numpy()
    predicted_words = []
    for index in np.argsort(np.squeeze(averaged_predictions))[:5]:
        predicted_words.append(index_to_word[index])
    return predicted_words

predict('The dog was sitting still', model)

```

And here is the output of that on our bad model:

```
['deserv', 'itll', 'materi', 'deed', 'virtu']
```
These don't seem to make much sense with the possible exception of The dog itll sit still. But we have all of the dimensions working correctly and should be able to focus on improvements to the model going forward.

### The embeddings

So now that we have trained a model, how do we get the embeddings?

```
embeddings = model.layers[0].weights[0].numpy()
embeddings.shape
```
This gives us an array of dimension (len(vocabulary), embedding_dim). So in this example (2336, 8)
So each of our words is represented by an 8d vector. What can we do with this?

First view them all in the [Tensorflow embeddings projector](https://projector.tensorflow.org/).

### Viewing the embeddings

We use this code to dump our embeddings and the labels to tsv. We can then load them into the projector.
```
with open('weights.tsv', 'wt') as f:
    np.savetxt(f, embeddings, delimiter='\t')

with open('labels.tsv', 'wt') as f:
    for a in range(len(index_to_word)):
        f.write(index_to_word[a]+'\t\n')
```

And this is what we get

![embeddings](/assets/tulkinghorn.png)

Although the detail given here does not make sense based on our model you can see that the projector is very useful. It illustrates the proximity of similar words to each other in vector space.

The complete code for this experiment can be found [here](https://github.com/anthonynolan/autocomplete-with-bleak-house).
