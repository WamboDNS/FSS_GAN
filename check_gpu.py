import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
print("Built with CUDA: " + str(tf.test.is_built_with_cuda()))
    
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)