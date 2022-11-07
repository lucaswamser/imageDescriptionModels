from models.model import Model
import tensorflow as tf
import numpy as np

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden,training=False):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class AttentionModel(Model):

    def __init__(self,vocab_size,max_length,embedding_dim,units=512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.units = units
        self.dataset = None
        self._load_model()


    def _load_model(self):
            input_feature = tf.keras.layers.Input(shape=(64, 2048,))
            fc = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(input_feature)
            input_state = tf.keras.layers.Input(shape=(self.units,))
            inputs_embeddings = tf.keras.layers.Input(shape=(1,))
            context_vector, attention_weights = BahdanauAttention(self.units)(fc, input_state)
            x =  tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)(inputs_embeddings)
            context_vector = tf.keras.layers.Reshape((1, context_vector.shape[1], ))(context_vector)
            x = tf.keras.layers.Concatenate(axis=-1)([context_vector, x])
            output, state = tf.keras.layers.GRU(self.units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')(x)
            x = tf.keras.layers.Dense(self.units)(output)
            x = tf.keras.layers.Reshape((-1, x.shape[2]))(x)
            x = tf.keras.layers.Dense(self.vocab_size, activation="softmax")(x)
            self.keras_model = tf.keras.Model(inputs=[input_feature, input_state, inputs_embeddings], outputs=[x,state,attention_weights])

                
    def compile(self):
      self.optimizer = tf.keras.optimizers.Adam()
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction='none')


      def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
      
      self.loss_function = loss_function

    def _data_generator(self,descriptions, photos, wordtoix, max_length, num_photos_per_batch):
        X, y = list(), list()
        n=0
        while 1:
            for key, desc_list in descriptions.items():
                n+=1
                photo = photos[key]
                for desc in desc_list:
                    seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                    X.append(photo)
                    seq_out = [0]*max_length
                    #print(seq_out)
                    for i,s in enumerate(seq):
                        seq_out[i] = s
                    y.append(seq_out)
                # yield the batch data
                if n==num_photos_per_batch:
                    yield np.array(X),  np.asarray(y, dtype=np.float32)
                    X, y = list(), list()
                    n=0

    def train(self,dataset,steps=10,epochs=100,batch_size=1):
          self.vocab_wordtoix = dataset.vocab_wordtoix
          self.vocab_ixtoword = dataset.vocab_ixtoword

          @tf.function
          def test_step(img_tensor, target):
            loss = 0
            hidden = tf.zeros((target.shape[0], 512))
            dec_input = tf.expand_dims([self.vocab_wordtoix['startseq']] * target.shape[0], 1)

            with tf.GradientTape() as tape:
              
                for i in range(1, target.shape[1]):
                  
                    predictions, hidden, _ = self.keras_model([img_tensor, hidden,dec_input])
                    loss += self.loss_function(target[:, i], predictions)
                    dec_input = tf.expand_dims(target[:, i], 1)
          
            total_loss = (loss / int(target.shape[1]))  
            return loss, total_loss  
                

          @tf.function
          def train_step(img_tensor, target):
            loss = 0
            hidden = tf.zeros((target.shape[0], 512))
            dec_input = tf.expand_dims([self.vocab_wordtoix['startseq']] * target.shape[0], 1)

            with tf.GradientTape() as tape:
              
                for i in range(1, target.shape[1]):
                  
                    predictions, hidden, _ = self.keras_model([img_tensor, hidden,dec_input])
                    loss += self.loss_function(target[:, i], predictions)
                    dec_input = tf.expand_dims(target[:, i], 1)

            total_loss = (loss / int(target.shape[1]))

            trainable_variables = self.keras_model.trainable_variables

            gradients = tape.gradient(loss, trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

            return loss, total_loss

          loss_plot = []
          num_steps = len(dataset.train_descriptions) // batch_size
          generator = self._data_generator(dataset.train_descriptions, dataset.encoding_train, dataset.vocab_wordtoix, dataset.vocab_max_length, batch_size)
          generator_test = self._data_generator(dataset.test_descriptions, dataset.test_train, dataset.vocab_wordtoix, dataset.vocab_max_length, batch_size)
          #print(max_length)
          for i,g in enumerate(generator):
            if i > epochs:
              break;
            batch_loss, total_loss = train_step(g[0],g[1])
            loss_plot.append(total_loss / num_steps)
            
            if (i % 100 == 0):
              average_batch_loss = batch_loss.numpy()/batch_size
              print(f'Epoch {i+1} Loss {average_batch_loss:.4f}')
              #print(i)
              #print(total_loss)
          
          return loss_plot
            
    def predict(self,img_tensor_val):
            attention_plot = np.zeros((self.max_length, 64))

            hidden = tf.zeros((1, self.units))
            #print(hidden)

            #temp_input = preprocess(image)
            img_tensor_val =  np.expand_dims(img_tensor_val,axis=0)
            #img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                        #-1,
                                                        #3img_tensor_val.shape[3]))
            features = img_tensor_val

            dec_input = tf.expand_dims([self.vocab_wordtoix['startseq']], 0)
            result = []
            #print(dec_input)
            for i in range(self.max_length):
                predictions, hidden, attention_weights =  self.keras_model([features,
                                                                hidden,dec_input])
                attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

                predicted_id = np.argmax(predictions)
                if predicted_id == 0:
                    return " ".join(result), attention_plot
                predicted_word = self.vocab_ixtoword[predicted_id]
                result.append(predicted_word)
                dec_input = tf.expand_dims([predicted_id], 0)

            attention_plot = attention_plot[:len(result), :]
            return " ".join(result), attention_plot
