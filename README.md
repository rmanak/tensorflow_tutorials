# ** WARNING ** : outdated materials, this material is pretty much outdated since the release of TF2.0+ 

# TensorFlow Tutorials/Examples

I have more materials here, but will wait until release of TF 2.0 and re-work these tutorials to make sure there are not using any depricated API.



These tutorials are all for **TensorFlow 1.8**. (See what is new in TF 1.9: [TensorFlow 1.9.0 Release](https://github.com/tensorflow/tensorflow/releases/tag/v1.9.0))

- Tutorial 1: Basic variables, placeholders, matrix operations and session in TF

- Tutorial 2: Batch gradient decent linear regression in TensorFlow

- Tutorial 3: Logical/Control operations

- Tutorial 4: Logistic regression using TensorFlow

- Tutorial 5: Fully connected neural net "from scratch" (using TF's matrix operations) for binary classification

- Tutorial 6: MNIST image classification using multilayer fully connected NN built using `tf.layers` API and data processing using `tf.data` API

- Tutorial 7: More on `tf.data.Dataset` API and building high performance IO and data processing pipeline

- Tutorial 8: Premade estimators and `tf.feature_column` API for classification of iris dataset.

- Tutorial 9: Writing custom estimators with `tf.estimator` API, CSV IO example with dataset API and `tf.summary` for tensorboard.

- Tutorial 10: Embedding layer to vectorize sequence of indicies (text vectorization) using `tf.nn.embedding_lookup`.

---

## Extras

These are mainly some confusing concepts in TF/Keras that I tried to clarify by code snippets.

- XTutorial 1: Demonstrates that `TimeDistributed(Dense(n))` layer in Keras is identical to applying `Dense(n)` layer.
