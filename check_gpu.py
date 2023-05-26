import tensorflow as tf
import sys

print(tf.config.list_physical_devices('GPU'))
print("Built with CUDA: " + str(tf.test.is_built_with_cuda()))
    
gpu = "/device:GPU:0"
if int(sys.argv[1]) == 1:
    gpu = "/device:GPU:1"
    
with tf.device(gpu):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print(c)