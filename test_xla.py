import tensorflow as tf
import time

from tensorflow import keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from model import model

layer = model.Darknet()


# @tf.function(experimental_compile=True)
# def call_func(input1, input2):
#     return tf.matmul(input1, input2) + input2 + (input1 * input1)


resnet = keras.applications.MobileNetV2(weights=None)
@tf.function(experimental_compile=True)
def call_func(inputs):
    return resnet(inputs)

# @tf.function()
# def call_func2(inputs):
#     return resnet(inputs)


a = tf.ones(shape=[10000, 10000])
b = a
inputs = tf.ones(shape=[1, 224, 224, 3])

for i in range(2):
    call_func(inputs)

start = time.time()
for i in range(10):
    res = call_func(inputs)
print(time.time() - start)

# start = time.time()
# for i in range(10):
#     res = call_func(a, b)
#     # res = resnet(inputs)
# print(time.time() - start)