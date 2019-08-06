import tensorflow as tf

class test(tf.Module):
    def __init__(self, name):
        super(test, self).__init__(name)
        self.w1 = tf.Variable(1)


a = test("sss")
print(a.trainable_variables)

