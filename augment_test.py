import tensorflow as tf

import augment


class AutoaugmentTest():

    def test_autoaugment_policy(self):
        # A very simple test to verify no syntax error.
        image = tf.cast(tf.random.uniform(shape=[800, 800, 3], dtype=tf.int32, maxval=255), tf.uint8)
        bboxes = tf.convert_to_tensor([[0, 0, 100, 200],
                                      [200, 200, 300, 500],
                                      [300, 400, 500, 600]], dtype=tf.float32) / 800.0
        # image = tf.placeholder(tf.uint8, shape=[640, 640, 3])
        # bboxes = tf.placeholder(tf.float32, shape=[4, 4])
        res = augment.distort_image_with_autoaugment(image, bboxes, 'v0')
        print(res)


if __name__ == '__main__':
    test = AutoaugmentTest()
    test.test_autoaugment_policy()
    # tf.test.main()
