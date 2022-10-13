import tensorflow as tf
import os
import shutil
import numpy as np

class GloveEmbeddingMatrix():
     def __init__(self,download_path,dataset,embedding_dim=200) -> None:
        self.download_path = download_path
        self.embedding_dim = embedding_dim
        self.dataset = dataset 
        self._download_glove()
        self._load()

     def _download_glove(self):
          out_file = os.path.join(self.download_path,"glove.6B.200d.txt")
          print(out_file)
          if not os.path.isfile(out_file):
              out_path = os.path.join(self.download_path, "glove.6B.zip")
              out_path_zip = os.path.join(self.download_path)
              tf.keras.utils.get_file(
                  fname=out_path,
                  origin="https://nlp.stanford.edu/data/glove.6B.zip",
              )
              shutil.unpack_archive(out_path, out_path_zip)
              os.remove(out_path)
          else:
              print("imagens existem")

     def _load(self):
        embeddings_index = {} # empty dictionary
        f = open(os.path.join(self.download_path, f'glove.6B.{self.embedding_dim}d.txt'), encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        # Get 200-dim dense vector for each of the 10000 words in out vocabulary
        self.embedding_matrix = np.zeros((self.dataset.vocab_size, self.embedding_dim))

        for word, i in self.dataset.vocab_wordtoix.items():
            #if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                 self.embedding_matrix[i] = embedding_vector
