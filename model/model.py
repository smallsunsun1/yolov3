import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from .util import broadcast_iou


def carafe(feature_map, cm, upsample_scale, k_encoder, kernel_size):
    """implementation os ICCV 2019 oral presentation CARAFE module"""
    static_shape = feature_map.get_shape().as_list()
    f1 = keras.layers.Conv2D(cm, (1, 1), padding="valid")(feature_map)
    encode_feature = keras.layers.Conv2D(upsample_scale * upsample_scale * kernel_size * kernel_size,
                                         (k_encoder, k_encoder), padding="same")(f1)
    encode_feature = tf.nn.depth_to_space(encode_feature, upsample_scale)
    encode_feature = tf.nn.softmax(encode_feature, axis=-1)
    """encode_feature [B x (h x scale) x (w x scale) x (kernel_size * kernel_size)]"""
    extract_feature = tf.image.extract_patches(feature_map, [1, kernel_size, kernel_size, 1],
                                               strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
    """extract feature [B x h x w x (channel x kernel_size x kernel_size)]"""
    extract_feature = keras.layers.UpSampling2D((upsample_scale, upsample_scale))(extract_feature)
    extract_feature_shape = tf.shape(extract_feature)
    B = extract_feature_shape[0]
    H = extract_feature_shape[1]
    W = extract_feature_shape[2]
    block_size = kernel_size * kernel_size
    extract_feature = tf.reshape(extract_feature, [B, H, W, block_size, -1])
    extract_feature = tf.transpose(extract_feature, [0, 1, 2, 4, 3])
    """extract feature [B x (h x scale) x (w x scale) x channel x (kernel_size x kernel_size)]"""
    encode_feature = tf.expand_dims(encode_feature, axis=-1)
    upsample_feature = tf.matmul(extract_feature, encode_feature)
    upsample_feature = tf.squeeze(upsample_feature, axis=-1)
    if static_shape[1] is None or static_shape[2] is None:
        upsample_feature.set_shape(static_shape)
    else:
        upsample_feature.set_shape(
            [static_shape[0], static_shape[1] * upsample_scale, static_shape[2] * upsample_scale, static_shape[3]])
    return upsample_feature


class DarknetConv(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding="same", use_gn=True, kernel_regularizer=None, **kwargs):
        super(DarknetConv, self).__init__(**kwargs)
        self.use_gn = use_gn
        self.conv = keras.layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=kernel_regularizer)
        self.gn = tfa.layers.GroupNormalization(32)
        self.relu = keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        if self.use_gn:
            x = self.gn(x)
        x = self.relu(x)
        return x


class DarknetResidual(keras.layers.Layer):
    def __init__(self, filters, strides=1, padding="same", kernel_regularizer=None, **kwargs):
        super(DarknetResidual, self).__init__(**kwargs)
        self.darknet_conv1 = DarknetConv(filters // 2, kernel_size=(1, 1), strides=strides, padding=padding,
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv2 = DarknetConv(filters, kernel_size=(3, 3), strides=strides, padding=padding,
                                         kernel_regularizer=kernel_regularizer)
        self.add = keras.layers.Add()

    def call(self, inputs, **kwargs):
        x = self.darknet_conv1(inputs)
        x = self.darknet_conv2(x)
        x = self.add([inputs, x])
        return x


class DarknetBlock(keras.layers.Layer):
    def __init__(self, filters, blocks, kernel_regularizer=None, **kwargs):
        super(DarknetBlock, self).__init__(**kwargs)
        self.darknet_conv = DarknetConv(filters, kernel_size=(3, 3), strides=(2, 2),
                                        kernel_regularizer=kernel_regularizer)
        self.seqence = keras.Sequential()
        for i in range(blocks):
            self.seqence.add(DarknetResidual(filters, **kwargs))

    def call(self, inputs, **kwargs):
        x = self.darknet_conv(inputs)
        x = self.seqence(x)
        return x


class Darknet(keras.layers.Layer):
    def __init__(self, kernel_regularizer=None, **kwargs):
        super(Darknet, self).__init__(**kwargs)
        self.darknet_conv1 = DarknetConv(32, (3, 3), (1, 1), **kwargs)
        self.darknet_block1 = DarknetBlock(64, 1, kernel_regularizer=kernel_regularizer)
        self.darknet_block2 = DarknetBlock(128, 2, kernel_regularizer=kernel_regularizer)
        self.darknet_block3 = DarknetBlock(256, 8, kernel_regularizer=kernel_regularizer)
        self.darknet_blcok4 = DarknetBlock(512, 8, kernel_regularizer=kernel_regularizer)
        self.darknet_block5 = DarknetBlock(1024, 4, kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        output = []
        x = self.darknet_conv1(inputs)
        x = self.darknet_block1(x)
        x = self.darknet_block2(x)
        x = self.darknet_block3(x)
        output.append(x)
        x = self.darknet_blcok4(x)
        output.append(x)
        x = self.darknet_block5(x)
        output.append(x)
        return output


class YoloConv(keras.layers.Layer):
    def __init__(self, filters, kernel_regularizer=None, **kwargs):
        super(YoloConv, self).__init__(**kwargs)
        self.darknet_conv1 = DarknetConv(filters, kernel_size=(1, 1), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv2 = DarknetConv(filters, kernel_size=(1, 1), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv3 = DarknetConv(filters * 2, kernel_size=(3, 3), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv4 = DarknetConv(filters, kernel_size=(1, 1), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv5 = DarknetConv(filters * 2, kernel_size=(3, 3), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv6 = DarknetConv(filters, kernel_size=(1, 1), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            x, x_skip = inputs
            x = self.darknet_conv1(x)
            x_shape = tf.shape(x)
            x = tf.image.resize(x, [2 * x_shape[1], 2 * x_shape[2]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.concat([x, x_skip], axis=-1)
        else:
            x = inputs
        x = self.darknet_conv2(x)
        x = self.darknet_conv3(x)
        x = self.darknet_conv4(x)
        x = self.darknet_conv5(x)
        x = self.darknet_conv6(x)
        return x


class YoloOutput(keras.layers.Layer):
    def __init__(self, filters, anchors, classes, kernel_regularizer=None, **kwargs):
        super(YoloOutput, self).__init__(**kwargs)
        self.anchors = anchors
        self.classes = classes
        self.darknet_conv1 = DarknetConv(filters * 2, kernel_size=(3, 3), strides=(1, 1),
                                         kernel_regularizer=kernel_regularizer)
        self.darknet_conv2 = DarknetConv(anchors * (classes + 5), (1, 1), (1, 1), use_gn=False,
                                         kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        x = self.darknet_conv1(inputs)
        x = self.darknet_conv2(x)
        x_shape = tf.shape(x)
        output = tf.reshape(x, [-1, x_shape[1], x_shape[2], self.anchors, self.classes + 5])
        return output


def yolo_boxes(pred, anchors, classes):
    """

    :param pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    :param anchors:  (num_anchors, 2)
    :param classes:
    :return:
    """
    grid_size_h = tf.shape(pred)[1]
    grid_size_w = tf.shape(pred)[2]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)
    box_xy = tf.sigmoid(box_xy)  # [batch_size, grid, grid, anchors, 2]
    objectness = tf.sigmoid(objectness)
    # class_probs = tf.sigmoid(class_probs)
    # class_probs = tf.nn.softmax(class_probs, axis=-1)
    pred_box = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss
    grid = tf.meshgrid(tf.range(grid_size_w), tf.range(grid_size_h))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    box_xy = (box_xy + tf.cast(grid, box_xy.dtype))
    box_wh = tf.exp(box_wh) * tf.cast(anchors, box_wh.dtype)
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, h, w):
    # boxes, conf, type
    b, c, t = [], [], []
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    bbox = tf.concat(b, axis=1)
    bbox = bbox * tf.cast(tf.concat([w, h, w, h], axis=-1), bbox.dtype)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.nn.softmax(tf.concat(t, axis=1), axis=-1)
    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.cast(tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)), tf.float32),
        scores=tf.cast(tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])), tf.float32),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5,
        clip_boxes=False
    )

    return boxes, scores, classes, valid_detections


class YoloV3(keras.layers.Layer):
    def __init__(self, anchors, masks, classes, kernel_regularizer=None, **kwargs):
        super(YoloV3, self).__init__(**kwargs)
        self.anchors = anchors
        self.masks = masks
        self.classes = classes
        self.darknet = Darknet(kernel_regularizer=kernel_regularizer)
        self.yolo_conv1 = YoloConv(512, kernel_regularizer=kernel_regularizer)
        self.yolo_conv2 = YoloConv(256, kernel_regularizer=kernel_regularizer)
        self.yolo_conv3 = YoloConv(128, kernel_regularizer=kernel_regularizer)
        self.yolo_out1 = YoloOutput(512, len(masks[0]), classes, kernel_regularizer=kernel_regularizer)
        self.yolo_out2 = YoloOutput(256, len(masks[1]), classes, kernel_regularizer=kernel_regularizer)
        self.yolo_out3 = YoloOutput(128, len(masks[2]), classes, kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        x, h, w = inputs
        size = tf.cast(tf.concat([h, w], axis=-1), tf.float32)
        x_36, x_61, x = self.darknet(x)
        x = self.yolo_conv1(x)
        output_0 = self.yolo_out1(x)
        x = self.yolo_conv2((x, x_61))
        output_1 = self.yolo_out2(x)
        x = self.yolo_conv3((x, x_36))
        output_2 = self.yolo_out3(x)
        anchor1 = self.anchors[self.masks[0]] / size
        anchor2 = self.anchors[self.masks[1]] / size
        anchor3 = self.anchors[self.masks[2]] / size
        boxes_0 = yolo_boxes(output_0,
                             anchor1,
                             self.classes)
        boxes_1 = yolo_boxes(output_1,
                             anchor2,
                             self.classes)
        boxes_2 = yolo_boxes(output_2,
                             anchor3,
                             self.classes)
        boxes, scores, classes, valid_detections = yolo_nms((boxes_0[:3], boxes_1[:3], boxes_2[:3]), h, w)
        return output_0, output_1, output_2, boxes, scores, classes, valid_detections


class YoloLoss(keras.losses.Loss):
    def __init__(self, anchors, classes=80, ignore_thresh=0.5, **kwargs):
        super(YoloLoss, self).__init__(**kwargs)
        self.anchors = anchors
        self.classes = classes
        self.ignore_thresh = ignore_thresh

    def call(self, y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, self.anchors, self.classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size_h = tf.shape(y_true)[1]
        grid_size_w = tf.shape(y_true)[2]
        grid = tf.meshgrid(tf.range(grid_size_w), tf.range(grid_size_h))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(tf.stack([grid_size_w, grid_size_h], axis=-1), true_xy.dtype) - \
                  tf.cast(grid, true_xy.dtype)
        h = tf.cast(grid_size_h * 32, tf.float32)
        w = tf.cast(grid_size_w * 32, tf.float32)
        true_wh = tf.math.log(true_wh / tf.cast(self.anchors / tf.stack([h, w], axis=-1), true_wh.dtype))
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)
        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            pred_box.dtype)
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, obj_mask.dtype)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
                   (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        # class_loss = obj_mask * keras.losses.binary_crossentropy(
        #     tf.one_hot(tf.cast(tf.squeeze(true_class_idx, axis=-1), tf.int32),
        #                depth=self.classes), pred_class, from_logits=False)
        class_loss = obj_mask * keras.losses.sparse_categorical_crossentropy(true_class_idx, pred_class,
                                                                             from_logits=True)
        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        # true_x1y1 = true_xy - true_wh / 2
        # true_x2y2 = true_xy + true_wh / 2
        # pred_x1y1 = pred_xy - pred_wh / 2
        # pred_x2y2 = pred_xy + pred_wh / 2
        # true_y1x1y2x2 = tf.concat([true_x1y1[..., ::-1], true_x2y2[..., ::-1]], axis=-1)
        # pred_y1x1y2x2 = tf.concat([pred_x1y1[..., ::-1], pred_x2y2[..., ::-1]], axis=-1)
        # boxes_loss = obj_mask * tfa.losses.giou_loss(true_y1x1y2x2, pred_y1x1y2x2) * box_loss_scale
        # boxes_loss = tf.reduce_sum(boxes_loss, axis=(1, 2, 3))
        xy_loss = tf.reduce_mean(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_mean(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        return tf.reduce_mean(xy_loss + wh_loss + obj_loss + class_loss)
        # xy_loss = tf.reduce_mean(xy_loss)
        # wh_loss = tf.reduce_mean(wh_loss)
        # obj_loss = tf.reduce_mean(obj_loss)
        # class_loss = tf.reduce_mean(class_loss)
        # return xy_loss + wh_loss + obj_loss + class_loss

