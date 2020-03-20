import tensorflow as tf
import yaml
import numpy as np
import re
import cv2

from tensorflow import keras

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from data_loader import input_fn
from model import model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

config_file = open("./config/yolov3.yaml")
params = yaml.load(config_file)
params["anchors"] = np.asarray(params["anchors"]) / np.asarray(params["image_size"])
params["masks"] = np.asarray(params["masks"])
anchors = params["anchors"]
masks = params["masks"]


def schedule(epoch):
    if epoch < 20:
        return 0.001
    elif epoch < 25:
        return 0.0001
    else:
        return 0.00001


image = keras.Input(shape=[None, None, 3], dtype=tf.float32, name='image')
key_prefix = "grids_{}"
yolov3 = model.YoloV3(params['anchors'], params['masks'], params['classes'], name='yolov3',
                      kernel_regularizer=None)
output_0, output_1, output_2, boxes, scores, classes, valid_detections = yolov3(image)
yolo_model = keras.Model(inputs=[image], outputs=[output_0, output_1, output_2,
                                                  boxes, scores, classes, valid_detections])
dataset = input_fn(params["train_files"],
                   params["anchors"], params["masks"],
                   params["classes"], params["image_size"],
                   params["batch_size"])
yolo_model.compile(optimizer=keras.optimizers.Adam(),
                   loss={'yolov3': model.YoloLoss(anchors[masks[0]], params["classes"]),
                         'yolov3_1': model.YoloLoss(anchors[masks[1]], params["classes"]),
                         'yolov3_2': model.YoloLoss(anchors[masks[2]], params["classes"])})
yolo_model.fit(dataset, steps_per_epoch=3000, epochs=30,
               callbacks=[
                   keras.callbacks.LearningRateScheduler(schedule, 1),
                   keras.callbacks.ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                                                   verbose=1, save_weights_only=True),
                   keras.callbacks.TensorBoard(log_dir='logs', update_freq=1000,
                                               profile_batch=2)]
               )
