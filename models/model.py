import keras


class Model(object):

    def __init__(self) -> None:
        pass

    def save_weights_to_file(self,file):
        self.keras_model.save_weights(file)
      
    def load_weights_from_file(self,file):
         self.keras_model.load_weights(file)

    def get_plot(self):
       return keras.utils.vis_utils.plot_model(self.keras_model, show_shapes=True, show_layer_names=True)

    def get_embedding_layer(self):
      for i,v in enumerate(self.keras_model.layers):
        if "embedding" in v.name:
         return i

    def set_glove(self,glove):
      embedding_layer = self.get_embedding_layer()
      print(f"setando o glove na layer {embedding_layer}")
      self.keras_model.layers[embedding_layer].set_weights([glove.embedding_matrix])
      self.keras_model.layers[embedding_layer].trainable = False

    

