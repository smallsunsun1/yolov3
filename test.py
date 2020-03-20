import tensorflow as tf
import numpy as np
import multiprocessing
from tensorflow import keras
import os
import re

tf.split()


if __name__ == "__main__":
    filename = "/home/admin-seu/hugh/yolov3-tf2/data_native/train.txt"
    info = []
    for ele in open(filename).readlines():
        ele = re.sub(",", " ", ele.strip())
        info.append(ele.split(" "))
    print(info)

