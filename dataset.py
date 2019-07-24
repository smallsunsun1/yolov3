import numpy as np
import tensorflow as tf



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
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

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


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train



# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('../../../Data/bear.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

def parse_image(image_paths, size):
    img = tf.map_fn(lambda x: tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3),(size, size)),
                     image_paths, dtype=tf.float32)
    # img = tf.map_fn(lambda x: tf.image.decode_jpeg(tf.io.read_file(x), channels=3), image_paths, dtype=tf.uint8)
    # img = tf.map_fn(lambda x: tf.image.resize(x, (size, size)), img, dtype=tf.float32)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)/ 255
    # img = tf.image.resize(img, (size, size)) / 255
    # print(img.shape)
    return img

def parse_bbox(bboxes, anchors, anchor_masks, num_class):
    label = tf.convert_to_tensor(bboxes, tf.float32)
    return transform_targets(bboxes, anchors, anchor_masks, num_class)

def parse_single_image(img_path, size, bbox):
    img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
    pad_bg = tf.ones((size,size,3))
    # h, w = tf.shape(img.shape)[:2]
    h, w = img.shape[:2]
    print(img)
    img = tf.image.resize_with_pad(img, size. size)
    ratio = size/max(h, w)



def load_mosaic_dataset(filename, size, batch_size, anchors, anchor_masks, num_class):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [d.strip().split(' ') for d in data]
    image_paths = [d[0] for d in data]
    bboxes = [np.array([box.split(',') for box in d[1:]], dtype=np.float32) for d in data]
    def pad(x):
        h = x.shape[0]
        if h == 5:
            return x
        else:
            n = 5 - h
            new_pad = np.pad(x, ((0, n),(0,0)), 'constant', constant_values=(0))
            return new_pad
    bboxes = [pad(box) for box in bboxes]
    # image_paths = tf.convert_to_tensor(image_paths)
    mosaic_dataset = tf.data.Dataset.from_tensor_slices((image_paths, bboxes))
    mosaic_dataset = mosaic_dataset.repeat()
    mosaic_dataset = mosaic_dataset.shuffle(buffer_size=10).batch(batch_size)
    mosaic_dataset = mosaic_dataset.map(lambda x, y: (
        parse_image(x, size),
        parse_bbox(y, anchors, anchor_masks, num_class)))
    mosaic_dataset = mosaic_dataset.prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)
    return mosaic_dataset

def fake_dataset():
    from models import yolo_anchors, yolo_anchor_masks
    filename = '/Users/junfenghe/Code/GitOA/Hughhe/YoloV3/data/test.txt'
    size = 418
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks
    num_class = 1
    batch_size = 1
    mosaic_dataset = load_fake_dataset()
    mosaic_dataset = mosaic_dataset.repeat(3)
    mosaic_dataset = mosaic_dataset.shuffle(buffer_size=10).batch(batch_size)
    mosaic_dataset = mosaic_dataset.map(lambda x, y: (
        transform_images(x, size),
        transform_targets(y, anchors, anchor_masks, num_class)))
    mosaic_dataset = mosaic_dataset.prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)
    for d, l in mosaic_dataset:
        print('BATCH SIZE', d.shape)
        print(*l)

def main():
    from models import yolo_anchors, yolo_anchor_masks
    filename = '/Users/junfenghe/Code/GitOA/Hughhe/YoloV3/data/test.txt'
    size = 418
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks
    num_class = 1
    batch_size = 5
    dataset = load_mosaic_dataset(filename, size, batch_size, anchors, anchor_masks, num_class)
    for d, l in dataset:
        print('BATCH SIZE', d.shape)
        print(l[0])
    # print(dataset)

if __name__ == '__main__':
    main()
    # # fake_dataset()
    # img_path = '/Users/junfenghe/Code/GitOA/Hughhe/YoloV3/output_dir/0.jpg'
    # parse_single_image(img_path, 416, (0,0,0,0,0))
