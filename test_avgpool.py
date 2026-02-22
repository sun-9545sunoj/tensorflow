import tensorflow as tf

# grad must be 5D for AvgPool3DGrad in NDHWC, but we pass 2D on purpose.
grad = tf.zeros([6, 2], dtype=tf.float32)

try:
    y = tf.raw_ops.AvgPool3DGrad(
        orig_input_shape=tf.constant([2, 3, 3, 3, 3], dtype=tf.int32),
        grad=grad,
        ksize=[1, 1, 1, 1, 1],
        strides=[1, 1, 1, 1, 1],
        padding="VALID",
        data_format="NDHWC",
    )
    print("Failed: did not raise expected error")
except tf.errors.InvalidArgumentError as e:
    print("Caught expected InvalidArgumentError:", e)
