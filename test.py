import tensorflow as tf
from tensorflow.contrib import distribute
import numpy as np
import multiprocessing
from tensorflow import keras

def test(a, *args):
    for i in range(len(args)):
        print(i)


if __name__ == "__main__":
    test(1, 2, 3, 4, 5)
    # tf.enable_eager_execution()
    # h = tf.range(5)
    # w = tf.range(4)
    # grid = tf.meshgrid(w, h)
    # grid = tf.stack(grid[::-1], axis=-1)
    # print(grid)

