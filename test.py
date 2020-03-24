import tensorflow as tf
import numpy as np
import multiprocessing
from tensorflow import keras
import os
import re


a = [[1], [2, 3, 4], [88]]
b = tf.ragged.constant(a)
c = tf.expand_dims(tf.range(1, tf.reduce_max(b)), axis=0)
d = tf.expand_dims(b.flat_values, axis=0)
set1 = tf.sets.difference(c, d)
set1 = tf.sparse.to_dense(set1)
num = tf.random.shuffle(tf.squeeze(set1, axis=0))[0]
print(b.to_tensor(num))
# tf.RaggedTensor.flat_values
tf.image.non_max_suppression()