import tensorflow as tf
from tensorflow import keras
from .util import broadcast_iou


def carafe(feature_map, cm, upsample_scale, k_encoder, kernel_size):
    """implementation os ICCV 2019 oral presentation CARAFE module"""
    f1 = keras.layers.Conv2D(cm, (1, 1), padding="valid")(feature_map)
    encode_feature = keras.layers.Conv2D(upsample_scale * upsample_scale * kernel_size * kernel_size,
                                         (k_encoder, k_encoder), padding="same")(f1)
    encode_feature = tf.nn.depth_to_space(encode_feature, upsample_scale)
    encode_feature = tf.nn.softmax(encode_feature, axis=-1)
    """encode_feature [B x (h x scale) x (w x scale) x (kernel_size * kernel_size)]"""
    extract_feature = tf.image.extract_image_patches(feature_map, [1, kernel_size, kernel_size, 1],
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
    return upsample_feature


def group_norm(x, G=32, esp=1e-5):
    x = tf.transpose(x, [0, 3, 1, 2])
    x_shape = tf.shape(x)
    N = x_shape[0]
    C = x_shape[1]
    H = x_shape[2]
    W = x_shape[3]
    G = tf.minimum(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = tf.Variable(tf.ones(shape=[C]), dtype=tf.float32, name='gamma')
    beta = tf.Variable(tf.ones(shape=[C]), dtype=tf.float32, name='beta')
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])

    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
    output = tf.transpose(output, [0, 2, 3, 1])
    return output


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = "same"
    else:
        x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        # x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"
    x = keras.layers.Conv2D(filters, size,
                         strides, padding)(x)
    if batch_norm:
        x = group_norm(x, 32)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = keras.layers.Add()([prev, x])
    return x

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def Darknet(inputs):
    x = DarknetConv(inputs, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2) # skip connection
    x = x_36 = DarknetBlock(x, 256, 8) # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return x_36, x_61, x


def YoloConv(inputs, filters):
    """

    :param inputs: tuple(Tensor, Tensor)
    :param filters:
    :return:
    """
    if isinstance(inputs, tuple):
        x, x_skip = inputs
        x = DarknetConv(x, filters, 1)
        x_shape = tf.shape(x)
#        x = tf.image.resize(x, [x_shape[1] * 2, x_shape[2] * 2])
        x = carafe(x, 3, 2, 3, 3)
        x = tf.concat([x, x_skip], axis=-1)
    else:
        x = inputs
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    return x


def YoloOutput(inputs, filters, anchors, classes):
    x = DarknetConv(inputs, filters * 2, 3)
    x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
    x_shape = tf.shape(x)
    output = tf.reshape(x, [-1, x_shape[1], x_shape[2], anchors, classes + 5])
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
    box_xy = tf.sigmoid(box_xy)   # [batch_size, grid, grid, anchors, 2]
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss
    grid = tf.meshgrid(tf.range(grid_size_w), tf.range(grid_size_h))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2) # [gx, gy, 1, 2]
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(tf.stack([grid_size_w, grid_size_h], axis=-1), tf.float32)
    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, classes, valid_detections


def YoloV3(inputs, anchors, masks, classes=80, training=False):
    x_36, x_61, x = Darknet(inputs)
    x = YoloConv(x, 512)
    output_0 = YoloOutput(x, 512, len(masks[0]), classes)
    x = YoloConv((x, x_61), 256)
    output_1 = YoloOutput(x, 256, len(masks[1]), classes)
    x = YoloConv((x, x_36), 128)
    output_2 = YoloOutput(x, 128, len(masks[2]), classes)
    if training:
        return output_0, output_1, output_2
    boxes_0 = yolo_boxes(output_0, anchors[masks[0]], classes)
    boxes_1 = yolo_boxes(output_1, anchors[masks[1]], classes)
    boxes_2 = yolo_boxes(output_2, anchors[masks[2]], classes)
    outputs = yolo_nms((boxes_0[:3], boxes_1[:3], boxes_2[:3]), anchors, masks, classes)
    return outputs


# def YoloLoss(y_true, y_pred, anchors, classes=80, ignore_thresh=0.5):
#     """
#     1. transform all pred outputs
#     # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, cls))
#     2. transform all true outputs
#     # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
#     :param y_true:
#     :param y_pred
#     :param anchors:
#     :param classes:
#     :param ignore_thresh:
#     :return:
#     """
#     pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
#     pred_xy = pred_xywh[..., 0:2]
#     pred_wh = pred_xywh[..., 2:4]
#     true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
#     true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
#     true_wh = true_box[..., 2:4] - true_box[..., 0:2]
#     # give higher weights to small boxes
#     box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
#
#     # 3. inverting the pred box equations
#     grid_size_h = tf.shape(y_true)[1]
#     grid_size_w = tf.shape(y_true)[2]
#     grid = tf.meshgrid(tf.range(grid_size_h), tf.range(grid_size_w))
#     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
#     grid_size_wh = tf.stack([grid_size_w, grid_size_h], axis=-1)
#     true_xy = true_xy * tf.cast(grid_size_wh, tf.float32) - tf.cast(grid, tf.float32)
#     true_wh_raw = true_wh
#     true_wh = tf.math.log(true_wh / anchors)
#     true_wh = tf.where(tf.equal(true_wh_raw, 0), tf.zeros_like(true_wh), true_wh)
#     # calculate all masks
#     obj_mask = tf.squeeze(true_obj, -1)
#     # ignore false positive when iou is over threshold
#     true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
#     best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
#     ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
#
#     # 5. calculate all losses
#     xy_loss = obj_mask * box_loss_scale * \
#               tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
#     wh_loss = obj_mask * box_loss_scale * \
#               tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
#     obj_loss = keras.losses.binary_crossentropy(true_obj, pred_obj)
#     obj_loss = obj_mask * obj_loss + \
#                (1 - obj_mask) * ignore_mask * obj_loss
#     class_loss = obj_mask * keras.losses.sparse_categorical_crossentropy(true_class_idx, pred_class)
#     xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
#     wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
#     obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
#     class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
#
#     return (xy_loss + wh_loss + obj_loss + class_loss) / 1000.0


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
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
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_wh_raw = true_wh
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)

        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.equal(true_wh_raw, 0),
                           tf.zeros_like(true_wh), true_wh)
        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(
            pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * keras.losses.sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return (xy_loss + wh_loss + obj_loss + class_loss) 
    return yolo_loss

# def yolo_boxes(pred, anchors, classes):
#     # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
#     grid_size = tf.shape(pred)[1]
#     box_xy, box_wh, objectness, class_probs = tf.split(
#         pred, (2, 2, 1, classes), axis=-1)
#
#     box_xy = tf.sigmoid(box_xy)
#     objectness = tf.sigmoid(objectness)
#     class_probs = tf.sigmoid(class_probs)
#     pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss
#
#     # !!! grid[x][y] == (y, x)
#     grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
#     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
#
#     box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
#         tf.cast(grid_size, tf.float32)
#     box_wh = tf.exp(box_wh) * anchors
#
#     box_x1y1 = box_xy - box_wh / 2
#     box_x2y2 = box_xy + box_wh / 2
#     bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
#
#     return bbox, objectness, class_probs, pred_box






