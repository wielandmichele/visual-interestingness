import tensorflow as tf

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return (-100*dy)
    return y, custom_grad

class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)

    def call(self, x):
        return grad_reverse(x)