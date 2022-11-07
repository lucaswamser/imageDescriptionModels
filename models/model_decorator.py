

from base64 import encode
import numpy as np
import tensorflow as tf
from keras import Model
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3

class ModelPredictDocorator(object):

    def __init__(self,model) -> None:
        self.model = model
        self.load_emb_model()

    def preprocess(self,image_path):
        # Convert all the images to size 299x299 as expected by the inception v3 model
        img = tf.keras.utils.load_img(image_path, target_size=(299, 299))
        # Convert PIL image to numpy array of 3-dimensions
        x = tf.keras.utils.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # preprocess the images using preprocess_input() from inception module
        x = preprocess_input(x)
        return x

    def encode(self,image):
      batch_features = self.image_features_extract_model( self.preprocess(image))
      batch_features  = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))
      return batch_features 

    def load_emb_model(self):
        image_model = InceptionV3(include_top=False,weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        self.image_features_extract_model = Model(new_input, hidden_layer)

    def predict(self,image_path):
        photo = self.encode(image_path)[0]
        return self.model.predict(photo)
