import numpy as np
import tensorflow as tf

class YoloKmeans(object):
    def __init__(self, cluster_number, filename="./2012_train.txt"):
        self.cluster_number = cluster_number
        self.filename = filename
    def iou(self, boxes, clusters): # box -> k cluster
        n = np.shape(boxes)[0]
