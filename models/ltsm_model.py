import keras
from models.model import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
class LTSMModel(Model):

    def __init__(self,vocab_size,max_length,embedding_dim) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.dataset = None
        self._load_model()


    def _load_model(self):
        inputs1 = tf.keras.layers.Input(shape=(64, 2048,))

        fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
        fe2 = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1))(fe1)

        #fe2 = tf.keras.layers.Dropout(0.5)(fe1)
        fe3 = tf.keras.layers.Dense(256, activation='relu')(fe2)
        inputs2 = tf.keras.layers.Input(shape=(self.max_length,))
        se1 = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)
        se2 = tf.keras.layers.Dropout(0.5)(se1)
        se3 = tf.keras.layers.LSTM(256)(se2)
        decoder1 = tf.keras.layers.add([fe3, se3])
        decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
        outputs = tf.keras.layers.Dense(self.vocab_size, activation='softmax')(decoder2)
        self.keras_model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)

    
    def get_plot(self):
       return keras.utils.vis_utils.plot_model(self.keras_model, show_shapes=True, show_layer_names=True)

    def _data_generator(self,descriptions, photos, wordtoix, max_length, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n=0
        while 1:
            for key, desc_list in descriptions.items():
                n+=1
                photo = photos[key]
                #print(photo)
                for desc in desc_list:
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq =  tf.keras.utils.pad_sequences([in_seq], maxlen=max_length)[0]
                        out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)
                # yield the batch data
                if n==num_photos_per_batch:
                    yield [[np.array(X1), np.array(X2)], np.array(y)]
                    X1, X2, y = list(), list(), list()
                    n=0

    def build(self):
        self.keras_model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self,dataset,steps=2000,epochs=10,batch_size=3):
        self.vocab_wordtoix = dataset.vocab_wordtoix
        self.vocab_ixtoword = dataset.vocab_ixtoword
        generator = self._data_generator(dataset.train_descriptions, dataset.encoding_train, dataset.vocab_wordtoix, dataset.vocab_max_length, batch_size)
        generator_test = self._data_generator(dataset.test_descriptions, dataset.encoding_test, dataset.vocab_wordtoix, dataset.vocab_max_length, batch_size)
        return self.keras_model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps, verbose=1,validation_data = generator_test)


    def predict(self,photo):
          photo = np.expand_dims(photo,axis=0)
          in_text = 'startseq'
          for i in range(self.max_length):
                sequence = [self.vocab_wordtoix[w] for w in in_text.split() if w in self.vocab_wordtoix]
                sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=self.max_length)
                yhat = self.keras_model.predict([photo,sequence], verbose=0)
                yhat = np.argmax(yhat)
                #print(yhat)
                word = self.vocab_ixtoword[yhat]
                in_text += ' ' + word
                if word == 'endseq':
                    break

          final = in_text.split()
          final = final[1:-1]
          final = ' '.join(final)
          return final

    def load_model_file(self,file):
        pass
