import tensorflow as tf
import yaml
import numpy as np
import cv2

from tensorflow import keras

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from data_loader import input_fn, test_input_fn
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

image = keras.Input(shape=[None, None, 3], dtype=tf.float32, name='image')
yolov3 = model.YoloV3(params['anchors'], params['masks'], params['classes'], name='yolov3',
                      kernel_regularizer=None)
output_0, output_1, output_2, boxes, scores, classes, valid_detections = yolov3(image)
yolo_model = keras.Model(inputs=[image], outputs=[output_0, output_1, output_2,
                                                  boxes, scores, classes, valid_detections])
yolo_model.load_weights('./checkpoints/yolov3_voc_7.ckpt')
filenames = []
for ele in open(params["test_files"][0]).readlines():
    filenames.append(ele.strip('\n'))


class DrawBoxCallBack(keras.callbacks.Callback):
    def __init__(self, thresh, outfile, update_frequency=500):
        super(DrawBoxCallBack, self).__init__()
        self.thresh = thresh
        self.writer = tf.summary.create_file_writer(outfile)
        self.update_frequency = update_frequency

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.update_frequency != 0:
            return
        total_image = []
        for idx in range(5):
            indices = int(np.random.uniform(0, len(filenames), []))
            img = cv2.imread(filenames[indices])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img - 127.5) / 255.0
            img_tensor = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 416, 416)
            _, _, _, boxes, scores, classes, valid_detections = yolo_model(img_tensor)
            image_data = np.squeeze(img_tensor.numpy(), axis=0)
            image_data = image_data * 255 + 127.5
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            for k in range(len(valid_detections)):
                for i in range(valid_detections[k]):
                    bbox = boxes[k][i]
                    cv2.rectangle(image_data, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (255, 0, 0), 1)
                    cv2.putText(image_data, "{:2.0f}".format(classes[k][i]), (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)
                    cv2.putText(image_data, "{:.2f}".format(scores[k][i]), (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
            cv2.imwrite("./result/test{}.jpg".format(idx), image_data)
            total_image.append(image_data)
            if idx == 4:
                break
        total_image = tf.cast(tf.clip_by_value(tf.convert_to_tensor(total_image), 0, 255), dtype=tf.uint8)
        with self.writer.as_default():
            tf.summary.image('detect_res', total_image, step=batch, max_outputs=5)


# test_dataset = test_input_fn(params['test_files'], 1)
# for idx, ele in enumerate(test_dataset):
#     _, _, _, boxes, scores, classes, valid_detections = yolo_model(ele['image'])
#     # _, _, _, boxes, scores, classes, valid_detections = yolo_model.predict(ele['image'])
#     # print(boxes)
#     print(scores)
#     # print(classes)
#     # print(valid_detections)
#     # boxes = boxes * 416
#     image_data = np.squeeze(ele['image'].numpy(), axis=0)
#     image_data = image_data * 255 + 127.5
#     image_data = np.clip(image_data, 0, 255).astype(np.uint8)
#     image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
#     for k in range(len(valid_detections)):
#         for i in range(valid_detections[k]):
#             bbox = boxes[k][i]
#             print(bbox)
#             # print(classes[k][i])
#             # print(scores[k][i])
#             cv2.rectangle(image_data, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
#                           (255, 0, 0), 1)
#             cv2.putText(image_data, "{:2.0f}".format(classes[k][i]), (bbox[0], bbox[1] - 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 0, 255), 1)
#             cv2.putText(image_data, "{:.2f}".format(scores[k][i]), (bbox[0], bbox[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 255, 0), 1)
#     cv2.imwrite('./result/{}.jpg'.format(idx), image_data)
#     if idx == 10:
#         break

dataset = input_fn(params["train_files"],
                   params["anchors"], params["masks"],
                   params["classes"], params["image_size"],
                   params["batch_size"])
val_dataset = input_fn(params["train_files"],
                       params["anchors"], params["masks"],
                       params["classes"], params["image_size"],
                       params["batch_size"], training=False)
yolo_model.compile(optimizer=keras.optimizers.Adam(),
                   loss={'yolov3': model.YoloLoss(anchors[masks[0]], params["classes"]),
                         'yolov3_1': model.YoloLoss(anchors[masks[1]], params["classes"]),
                         'yolov3_2': model.YoloLoss(anchors[masks[2]], params["classes"])},
                   loss_weights={'yolov3': 1,
                                 'yolov3_1': 1,
                                 'yolov3_2': 1})


def schedule(epoch):
    if epoch < 45:
        return 0.001
    elif epoch < 55:
        return 0.0001
    else:
        return 0.00001


yolo_model.fit(dataset, steps_per_epoch=3000, epochs=60,
               callbacks=[
                   keras.callbacks.LearningRateScheduler(schedule=schedule),
                   keras.callbacks.ModelCheckpoint('checkpoints/yolov3_voc_{epoch}.ckpt',
                                                   verbose=1, save_weights_only=True),
                   keras.callbacks.TensorBoard(log_dir='logs', update_freq=100,
                                               profile_batch=0),
                   DrawBoxCallBack(0.5, './logs')
               ],
               validation_data=val_dataset, validation_steps=1000
               )
