import tensorflow as tf
import yaml
import numpy as np
import cv2


from tensorflow import keras
from data_loader import input_fn, input_fn_v2, test_input_fn


from model import model

tf.logging.set_verbosity(tf.logging.INFO)


def yolov3_model(features, labels, mode, params):
    anchors = params["anchors"]
    masks = params["masks"]
    classes = params["classes"]
    lr = params["learning_rate"]
    is_train = (mode != tf.estimator.ModeKeys.PREDICT)
    image = features["image"]
    image_size = params["image_size"]
    image.set_shape([None, None, None, 3])
    # image.set_shape([params["batch_size"], image_size[1], image_size[0], 3])
    key_prefix = "grids_{}"
    if is_train:
        output_0, output_1, output_2 = model.YoloV3(image, anchors, masks, classes, is_train)
        outputs = []
        outputs.append(output_0)
        outputs.append(output_1)
        outputs.append(output_2)
        print("output_0 shape is xxxxx ", output_0.shape)
        print("output_1 shape is xxxxx ", output_1.shape)
        print("output_2 shape is xxxxx ", output_2.shape)
    else:
        outputs = model.YoloV3(image, anchors, masks, classes, is_train)
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = []
        pred_loss = []
        loss = [model.YoloLoss(anchors[mask], classes) for mask in masks]
        for i in range(len(masks)):
            labels.append(features[key_prefix.format(i)])
        for output, label, loss_fn in zip(outputs, labels, loss):
            pred_loss.append(loss_fn(label, output))
        total_loss = tf.reduce_sum(pred_loss)
        # for i in range(len(masks)):
        #     loss = model.YoloLoss(features[key_prefix.format(i)], outputs[i], anchors[masks[i]], classes)
        #     pred_loss.append(loss)
        #     total_loss += tf.reduce_sum(pred_loss)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # ops = tf.get_default_graph().get_operations()
            # update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSubVariableOp")]
            # for op in update_ops:
            #     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # print(update_ops)

            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            # with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, tf.compat.v1.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=total_loss)
    else:
        prediction = {"image": image,
                      "boxes": outputs[0],
                      "scores": outputs[1],
                      "classes": outputs[2],
                      "valid_detections": outputs[3]}
        return tf.estimator.EstimatorSpec(mode, prediction)



if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    config_file = open("./config/yolov3.yaml")
    config = yaml.load(config_file)
    config["anchors"] = np.asarray(config["anchors"]) / np.asarray(config["image_size"])
    config["masks"] = np.asarray(config["masks"])
    num_gpus = config["gpus"]
    # session_configs = tf.ConfigProto(allow_soft_placement=True)
    # session_configs.gpu_options.allow_growth = True
    Config = tf.estimator.RunConfig(train_distribute=strategy,
                                    log_step_count_steps=100, save_checkpoints_steps=2000,
                                    eval_distribute=strategy, save_summary_steps=500)
    estimator = tf.estimator.Estimator(model_fn=yolov3_model, model_dir=config["model_dir"],
                                       config=Config, params=config)
    train_spec= tf.estimator.TrainSpec(input_fn=lambda :input_fn(config["train_files"],
                                                                 config["anchors"], config["masks"],
                                                                 config["classes"], config["image_size"],
                                                                 config["batch_size"]), max_steps=200000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda :input_fn(config["eval_files"],
                                                                 config["anchors"], config["masks"],
                                                                 config["classes"], config["image_size"],
                                                                 config["batch_size"], False), steps=None, throttle_secs=100)
#    tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
#     res = estimator.predict(input_fn=lambda :input_fn(config["test_files"],
#                                                                  config["anchors"], config["masks"],
#                                                                  config["classes"], config["image_size"],
#                                                                  config["batch_size"], False))
    res = estimator.predict(input_fn=lambda: test_input_fn(config["test_files"],
                                                      config["batch_size"], config["image_size"]))
    index = 0
    for ele in res:
        num_detections = ele['valid_detections']
        boxes = ele["boxes"][:num_detections]
        scores = ele["scores"][:num_detections]
        classes = ele["classes"][:num_detections]
        image = ele["image"]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        h, w = np.shape(image)[:2]
        for i in range(num_detections):
            x1 = int(boxes[i, 0] * w)
            x2 = int(boxes[i, 2] * w)
            y1 = int(boxes[i, 1] * h)
            y2 = int(boxes[i, 3] * h)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./result/{}.jpg".format(index), image)
        index += 1
        if index == 100:
            break

    """
    is_train = True
    eager = False
    params = config
    anchors = params["anchors"]
    masks = params["masks"]
    classes = params["classes"]
    lr = params["learning_rate"]
    image_size = params["image_size"]
    key_prefix = "grids_{}"
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input = keras.Input([None, None, 3])
        output_0, output_1, output_2 = model.YoloV3(input, anchors, masks, classes, is_train)
        yolo_model = keras.Model(inputs=input, outputs=[output_0, output_1, output_2])
        optimizer = tf.keras.optimizers.Adam(lr=params["learning_rate"])
        loss = [model.YoloLoss(anchors[mask], classes) for mask in masks]
        yolo_model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)
    filename = "/home/admin-seu/hugh/yolov3-tf2/data_native/train.txt"
    info = []
    for ele in open(filename).readlines():
        ele = re.sub(",", " ", ele.strip())
        info.append(ele.split(" "))
    dataset = input_fn_v2(tf.convert_to_tensor(info),
                       config["anchors"], config["masks"],
                       config["classes"], config["image_size"],
                       config["batch_size"])
    eval_dataset = input_fn_v2(tf.convert_to_tensor(info),
                            config["anchors"], config["masks"],
                            config["classes"], config["image_size"],
                            config["batch_size"])
    # for ele in dataset:
    #     print(ele)
    if eager:
        for features in dataset:
            with tf.GradientTape() as tape:
                image = features["image"]
                image.set_shape([params["batch_size"], image_size[1], image_size[0], 3])
                if is_train:
                    o0, o1, o2 = yolo_model(image, is_train)
                    outputs = []
                    outputs.append(o0)
                    outputs.append(o1)
                    outputs.append(o2)
                    labels = []
                    pred_loss = []
                    for i in range(len(masks)):
                        labels.append(features[key_prefix.format(i)])
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss)
                    print(total_loss)
                    grads = tape.gradient(total_loss, yolo_model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, yolo_model.trainable_variables))
    else:
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, verbose=1),
            keras.callbacks.ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            keras.callbacks.TensorBoard(log_dir='logs')
        ]
        history = yolo_model.fit(dataset,
                            epochs=5,
                            callbacks=callbacks,)
        print(history)
    """

