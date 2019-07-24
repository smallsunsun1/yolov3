import tensorflow as tf
import numpy as np
import cv2
import os


def transform_img_and_boxes(lines, target_size=(416, 416)):
    """

    :param imagename: 图片文件路径
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
    h_scale = tf.cast(tf.divide(target_h, img_h), tf.float32)
    w_scale = tf.cast(tf.divide(target_w, img_w), tf.float32)
    scale = tf.minimum(h_scale, w_scale)
    new_h = tf.cast(tf.multiply(tf.cast(img_h, tf.float32), scale), tf.int32)
    new_w = tf.cast(tf.multiply(tf.cast(img_w, tf.float32), scale), tf.int32)
    pad_h_top = (target_h - new_h) // 2
    pad_h_bottom = target_h - new_h - pad_h_top
    pad_w_left = (target_w - new_w) // 2
    pad_w_right = target_w - new_w - pad_w_left
    image = tf.squeeze(tf.image.resize(tf.expand_dims(image, axis=0), [new_h, new_w]), axis=0)
    image = tf.pad(image, [[pad_h_top, pad_h_bottom], [pad_w_left, pad_w_right], [0, 0]])
    box_l = (boxes[:, 0] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w
    box_r = (boxes[:, 2] * tf.cast(img_w, tf.float32) * scale + tf.cast(pad_w_left, tf.float32)) / target_w
    box_t = (boxes[:, 1] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h
    box_b = (boxes[:, 3] * tf.cast(img_h, tf.float32) * scale + tf.cast(pad_h_top, tf.float32)) / target_h
    boxes = tf.stack([box_l, box_t, box_r, box_b, boxes[:, 4]], axis=1)
    return image, boxes


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            # if tf.equal(area(y_true[i][j][:4]), 0):
            #     continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def split_line(lines):
    lines = tf.strings.regex_replace(lines, ",", " ")
    lines = tf.strings.split(lines, " ")
    lines = lines.to_tensor("0")
    # lines = tf.sparse.to_dense(lines.indices, lines.dense_shape, lines.values, default_value="0")
    return lines


def parse_lines(lines, target_size):
    images, bboxes = tf.map_fn(lambda x: transform_img_and_boxes(x, target_size), elems=lines,
                               dtype=(tf.float32, tf.float32), parallel_iterations=4, infer_shape=False)
    images.set_shape([None, target_size[0], target_size[1], 3])
    return images, bboxes


def area(boxes):
    return (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])


# def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
#     # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
#     N = tf.shape(y_true)[0]
#     M = tf.shape(y_true)[1]
#     if not isinstance(grid_size, list) and not isinstance(grid_size, tuple):
#         grid_size = [grid_size, grid_size]
#     # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
#     y_true_out = tf.zeros((N, grid_size[1], grid_size[0], tf.shape(anchor_idxs)[0], 6), dtype=tf.float32)
#
#     anchor_idxs = tf.cast(anchor_idxs, tf.int32)
#
#     indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True, infer_shape=True)
#     updates = tf.TensorArray(tf.float32, 0, dynamic_size=True, infer_shape=True)
#     idx = 0
#
#     def cond(i, j, n, m, idx, indexes, updates):
#         return i < n
#
#     def body(i, j, n, m, idx, indexes, updates):
#         anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
#         box = y_true[i][j][0:4]
#         box_area = area(box)
#         condition = tf.logical_and(tf.not_equal(box_area, 0), tf.reduce_any(anchor_eq))
#         def true_fn(indexes, updates, idx):
#             box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2.0
#             anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
#             grid_xy = tf.cast(box_xy * grid_size, tf.int32)
#             indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
#             updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
#             idx += 1
#             return idx, indexes, updates
#
#         def false_fn(indexes, updates, idx):
#             return idx, indexes, updates
#
#         idx, indexes, updates = tf.cond(condition, lambda: true_fn(indexes, updates, idx),
#                                         lambda: false_fn(indexes, updates, idx))
#         j += 1
#         j = tf.mod(j, m)
#         i = tf.cond(tf.equal(j, 0), lambda: i + 1, lambda: i)
#         return i, j, n, m, idx, indexes, updates
#
#     *_, idx, indexes, updates = tf.while_loop(cond, body, loop_vars=[0, 0, N, M, idx, indexes, updates])
#
#     y_true_out = y_true_out + tf.scatter_nd(indexes.stack(), updates.stack(), tf.shape(y_true_out))
#     return y_true_out

def transform_targets(y_train, anchors, anchor_masks, classes):
    y_outs = []
    grid_size = 13

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                   tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs, classes))
        grid_size *= 2

    return tuple(y_outs)


def wrap_dict(image, args):
    features = {}
    features['image'] = image
    for i in range(len(args)):
        features["grids_{}".format(i)] = args[i]
    return features


def input_fn(filenames, anchors, anchor_masks, classes, target_size=(416, 416), batch_size=4, pad_box_length=20):
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(split_line, num_parallel_calls=4)
    dataset = dataset.map(lambda x: parse_lines(x, target_size), 4)
    dataset = dataset.map(lambda x, y: (x / 255.0, transform_targets(y, anchors, anchor_masks, classes)), 4)
    dataset = dataset.map(wrap_dict, 4)
    dataset = dataset.prefetch(-1)
    return dataset


if __name__ == "__main__":
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / 416
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    filename = "/home/admin-seu/hugh/yolov3-tf2/data_native/train.txt"
    output_dir = "/home/admin-seu/hugh/yolov3-tf2/temp_file"
    dataset = input_fn(filename, yolo_anchors, yolo_anchor_masks, 2, batch_size=4)
    for idx, ele in enumerate(dataset):
        print(tf.where(tf.equal(ele["grids_0"][:, :, :, :, 4], 1)))
        print(tf.where(tf.equal(ele["grids_1"][:, :, :, :, 4], 1)))
        print(tf.where(tf.equal(ele["grids_2"][:, :, :, :, 4], 1)))
        print("stop")
        # image = tf.cast(ele[0], tf.uint8).numpy()[0]
        # boxes = ele[1]
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
        if idx == 0:
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
