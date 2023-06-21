import tensorflow as tf
from tensorflow.python.keras.layers import LeakyReLU

'''
    Models used by the GAN.
'''

# TODO: Change units

class Generator(tf.keras.Sequential):
    def __init__(self, layers=None, name=None, units=100):
        super().__init__(layers, name)
        self.add(tf.keras.layers.Dense(units))
        self.add(LeakyReLU())
        self.add(tf.keras.layers.Dense(units))
        self.add(LeakyReLU())
        print(self.layers)
        
class Discriminator(tf.keras.Sequential):
    def __init__(self, layers=None, name=None, units=100):
        super().__init__(layers, name)
        self.add(tf.keras.layers.Dense(units))
        self.add(LeakyReLU())
        self.add(tf.keras.layers.Dense(units))
        self.add(LeakyReLU())
        print(self.layers)
