# Custom L1 distance layer module


# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer


#Custom L1 Distance Layer from jupyter
class L1Dist(Layer):
    
    #Init method for inheritance from keras layers
    def __init__(self, **kwargs):
        super().__init__()
        
    #Function which tells what to do when input is passed
    #Similarity calculation is done
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)