import tensorflow as tf
import numpy as np
import cv2

import augment


def read_and_scale_image(path, target_size):
    content = tf.io.read_file(path)
    image = tf.image.decode_image(content, channels=3)
    image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    image = (image - 127.5) / 255.0
    return image


@tf.function
def transform_img_and_boxes(lines, target_size, training=True):
    """
    :param imagename:
    :param boxes: [N x 4]  x0, y0, x1, y1
    :param target_size:
    :return:
    """
    imagename = lines[0]
    boxes = tf.strings.to_number(lines[1:])
    boxes = tf.reshape(boxes, [-1, 5])
    content = tf.io.read_file(imagename)
    image = tf.image.decode_image(content, channels=3)
    image.set_shape([None, None, 3])
    target_h = target_size[0]
    target_w = target_size[1]
    image_shape = tf.shape(image)
    img_h = image_shape[0]
    img_w = image_shape[1]
    h_scale = tf.cast(target_h, tf.float32) / tf.cast(img_h, tf.float32)
    w_scale = tf.cast(target_w, tf.float32) / tf.cast(img_w, tf.float32)
    scale = tf.minimum(h_scale, w_scale)
    new_h = tf.cast(tf.multiply(tf.cast(img_h, tf.float32), scale), tf.int32)
    new_w = tf.cast(tf.multiply(tf.cast(img_w, tf.float32), scale), tf.int32)
    pad_h_top = (target_h - new_h) // 2
    # pad_h_bottom = target_h - new_h - pad_h_top
    pad_w_left = (target_w - new_w) // 2
    # pad_w_right = target_w - new_w - pad_w_left
    image = tf.image.resize_with_pad(image, target_h, target_w)
    image = tf.cast(image, tf.uint8)
    # image = tf.cast(tf.image.resize(image, [target_h, target_w]), tf.uint8)
    # image = tf.pad(image, [[pad_h_top, pad_h_bottom], [pad_w_left, pad_w_right], [0, 0]])
    target_w = tf.cast(target_w, tf.float32)
    target_h = tf.cast(target_h, tf.float32)
    box_l = (boxes[:, 0] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w
    box_r = (boxes[:, 2] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w
    box_t = (boxes[:, 1] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h
    box_b = (boxes[:, 3] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h
    # coordinate = tf.stack([box_t, box_l, box_b, box_r], axis=-1)
    # coordinate = tf.clip_by_value(coordinate, 0, 1)
    if training:
        # image, coordinate = augment.distort_image_with_autoaugment(image, coordinate, 'v0')
        p1 = tf.random.uniform([], 0, 10)
        p2 = tf.random.uniform([], 0, 10)
        p_b = tf.random.uniform([], 0, 10)
        p_c = tf.random.uniform([], 0, 10)
        p_h = tf.random.uniform([], 0, 10)
        if tf.greater(p_b, 5):
            image = tf.image.random_brightness(image, 0.1)
        if tf.greater(p_c, 5):
            image = tf.image.random_contrast(image, 0.1, 0.2)
        if tf.greater(p_h, 5):
            image = tf.image.random_hue(image, 0.1)
        image = tf.clip_by_value(tf.cast(image, tf.float32), 0, 255)
        cond1 = tf.greater(p1, 5.0)
        cond2 = tf.greater(p2, 5.0)

        def flip_left_right(image, box_l, box_r, box_t, box_b):
            image = tf.image.flip_left_right(image)
            box_l = 1.0 - box_l
            box_r = 1.0 - box_r
            return image, box_r, box_l, box_t, box_b

        def flip_top_down(image, box_l, box_r, box_t, box_b):
            image = tf.image.flip_up_down(image)
            box_t = 1.0 - box_t
            box_b = 1.0 - box_b
            return image, box_l, box_r, box_b, box_t

        image, box_l, box_r, box_t, box_b = tf.cond(cond1, lambda: flip_left_right(image, box_l, box_r, box_t, box_b),
                                                    lambda: (image, box_l, box_r, box_t, box_b))
        image, box_l, box_r, box_t, box_b = tf.cond(cond2, lambda: flip_top_down(image, box_l, box_r, box_t, box_b),
                                                    lambda: (image, box_l, box_r, box_t, box_b))
    # boxes = tf.stack([coordinate[:, 1], coordinate[:, 0], coordinate[:, 3], coordinate[:, 2], boxes[:, 4]], axis=-1)
    image.set_shape([None, None, 3])
    boxes = tf.stack([box_l, box_t, box_r, box_b, boxes[:, 4]], axis=1)
    return tf.cast(image, tf.float32), boxes


@tf.function
def transform_targets_for_output(y_true, grid_size_h, grid_size_w, anchor_idxs, h, w):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]
    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, tf.cast(grid_size_h, tf.int32), tf.cast(grid_size_w, tf.int32), tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False, element_shape=[4, ])
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False, element_shape=[6, ])
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                box_xy = tf.clip_by_value(box_xy, 0.0, 0.99999)
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / tf.stack([grid_size_w, grid_size_h], axis=-1)), tf.int32)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def split_line(lines):
    lines = tf.strings.regex_replace(lines, ",", " ")
    lines = tf.strings.split(lines, " ")
    lines = lines.to_tensor("0")
    return lines


def parse_lines(lines, target_size, training=True):
    size = tf.random.uniform([], 416, 609, dtype=tf.int32)
    size = tf.cast(tf.multiply(tf.cast(tf.divide(size, 32), tf.int32), 32), tf.int32)
    target_size = [size, size]
    images, bboxes = tf.map_fn(lambda x: transform_img_and_boxes(x, target_size, training), elems=lines,
                               dtype=(tf.float32, tf.float32),
                               parallel_iterations=4, infer_shape=False)
    return images, bboxes, tf.cast(size, tf.float32), tf.cast(size, tf.float32)


def area(boxes):
    return (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])


def transform_targets(y_train, h, w, anchors, anchor_masks):
    ##h, w 表示图形高和宽
    y_outs = []
    grid_size_h = h // 32
    grid_size_w = w // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32) / (tf.cast(tf.stack([w, h], axis=-1), tf.float32))
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size_h, grid_size_w, anchor_idxs, h, w))
        grid_size_h *= 2
        grid_size_w *= 2

    return tuple(y_outs)


def wrap_dict(image, args, h, w, batch_size):
    dict1 = {}
    image.set_shape([batch_size, None, None, 3])
    dict1['image'] = image
    dict1['h'] = tf.expand_dims(h, axis=0)
    dict1['w'] = tf.expand_dims(w, axis=0)
    features = {}
    for i in range(len(args)):
        if i == 0:
            features['yolov3'] = args[i]
        else:
            features["yolov3_{}".format(i)] = args[i]
    return (dict1, features)


def scale(x):
    return (x - 127.5) / 255.0


# @tf.function
def input_fn(filenames, anchors, anchor_masks, batch_size=4, training=True):
    dataset = tf.data.TextLineDataset(filenames)
    if training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(split_line, num_parallel_calls=8)
    dataset = dataset.map(lambda x: parse_lines(x, training), 8)
    dataset = dataset.map(lambda x, y, h, w: (scale(x), h, w, transform_targets(y, h, w, anchors, anchor_masks)), 8)
    dataset = dataset.map(lambda x, h, w, y: wrap_dict(x, y, h, w, batch_size), 8)
    dataset = dataset.prefetch(-1)
    return dataset


def test_input_fn(filenames, batch_size=4, target_size=(608, 608)):
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(lambda x: read_and_scale_image(x, target_size), 4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: {"image": x,
                                     "h": tf.expand_dims(target_size[0], axis=-1),
                                     "w": tf.expand_dims(target_size[1], axis=-1)}, 4)
    dataset = dataset.prefetch(-1)
    return dataset


if __name__ == "__main__":
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32)
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    filename = "/home/sunjiahe/Datasets/VOCDataset/yolo_train.txt"
    # output_dir = "/home/admin-seu/hugh/yolov3-tf2/temp_file"
    dataset = input_fn(filename, yolo_anchors, yolo_anchor_masks, batch_size=1)
    for idx, ele in enumerate(dataset):
        print(ele[0]['image'])
        image = ele[0]['image'].numpy()
        cv2.imwrite('./result/{}.jpg'.format(idx), cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB))
        # data = tf.reduce_max(ele[1]['yolov3'], axis=-1)
        # indices = tf.where(tf.not_equal(data, 0))
        # print(tf.gather_nd(data, indices))
        # print(ele[1])
        # print(tf.where(tf.equal(ele["grids_0"][:, :, :, :, 4], 1)))
        # print(tf.where(tf.equal(ele["grids_1"][:, :, :, :, 4], 1)))
        # print(tf.where(tf.equal(ele["grids_2"][:, :, :, :, 4], 1)))
        # print("stop")
        # image = tf.cast(ele[0], tf.uint8).numpy()[0]
        # boxes = ele[1]e
        # print(ele)

        # print(boxes)
        # if not list(tf.shape(boxes).numpy()) == [4, 1, 5]:
        #     print("error")
        # for index, value in enumerate(boxes):
        #     # print(value)
        #     l = int(value[0].numpy() * 416)
        #     r = int(value[2].numpy()  * 416)
        #     t = int(value[1].numpy()  * 416)
        #     b = int(value[3].numpy()  * 416)
        #     print((l, t), (r, b))
        #     image = cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)
        # cv2.imwrite(os.path.join(output_dir, "{}.jpg".format(idx)), image)
        if idx == 10:
            break

        # print(ele[1])
        # print(tf.where(tf.greater(ele[1], 1)))
    # content = tf.io.read_file("/Users/sunjiahe/PycharmProjects/yolo-V3/data/train/2.jpg")
    # image = tf.image.decode_image(content, channels=3)
    # boxes = tf.convert_to_tensor([[1, 1, 100, 100],
    #                               [2, 2, 200, 200]])
    # out, b = transform_img_and_boxes(image, boxes, (400, 400))
    # out = tf.stack([out[:,:,2], out[:,:,1], out[:,:,0]], axis=2)
    # # cv2.imshow("win", out.numpy().astype(np.uint8))
    # # cv2.waitKey()
    # print(b)
