import os
import tensorflow as tf
import shutil
import string
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import numpy as np
import pickle
class Dataset():

    def __init__(self, download_path) -> None:
        self.download_path = download_path
        pass


class Flickr8kDataset(Dataset):

    def __init__(self, download_path) -> None:
        super().__init__(download_path)
        self._download()
        self._load()

    def load_doc(self, filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text

    def load_descriptions(self,doc):
        mapping = dict()
        # process lines
        for line in doc.split('\n'):
            # split line by white space
            tokens = line.split()
            if len(line) < 2:
                continue
            # take the first token as the image id, the rest as the description
            image_id, image_desc = tokens[0], tokens[1:]
            # extract filename from image id
            image_id = image_id.split('.')[0]
            # convert description tokens back to string
            image_desc = ' '.join(image_desc)
            # create the list if needed
            if image_id not in mapping:
                mapping[image_id] = list()
            # store description
            mapping[image_id].append(image_desc)
        return mapping

    def clean_descriptions(self,descriptions):
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in descriptions.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                # tokenize
                desc = desc.split()
                # convert to lower case
                desc = [word.lower() for word in desc]
                # remove punctuation from each token
                desc = [w.translate(table) for w in desc]
                # remove hanging 's' and 'a'
                desc = [word for word in desc if len(word) > 1]
                # remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # store as string
                desc_list[i] = ' '.join(desc)

    def to_vocabulary(self,descriptions):
        # build a list of all description strings
        all_desc = set()
        for key in descriptions.keys():
            [all_desc.update(d.split()) for d in descriptions[key]]
        return all_desc

    def load_set(self,filename):
        doc = self.load_doc(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
            # skip empty lines
            if len(line) < 1:
                continue
            # get the image identifier
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)

        # load clean descriptions into memory
    def load_clean_descriptions(self,filename, dataset):
        # load document
        doc = self.load_doc(filename)
        descriptions = dict()
        for line in doc.split('\n'):
            # split line by white space
            tokens = line.split()
            # split id from description
            image_id, image_desc = tokens[0], tokens[1:]
            # skip images not in the set
            if image_id in dataset:
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # wrap description in tokens
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                # store
                descriptions[image_id].append(desc)
        return descriptions
      
    def save_descriptions(self,descriptions, filename):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()
    


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

    def encode(self,image_features_extract_model,image):
      batch_features = image_features_extract_model( self.preprocess(image))
      batch_features  = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))
      return batch_features 

    def _load_images_names(self):
        out_encoded_train_file = os.path.join(self.download_path,"encoded_train_images.pkl")
        out_encoded_test_file = os.path.join(self.download_path,"encoded_test_images.pkl")

        if (os.path.isfile(out_encoded_train_file)):
          print("carregando embs das imagens, já existente")
          self.encoding_train = pickle.load(open(out_encoded_train_file, "rb"))
          self.encoding_test = pickle.load(open(out_encoded_test_file, "rb"))
        else:
          print("baixando imagens")
          self._download_images()
          image_model = InceptionV3(include_top=False,weights='imagenet')
          new_input = image_model.input
          hidden_layer = image_model.layers[-1].output
          image_features_extract_model = Model(new_input, hidden_layer)


          train_images =  list(set(open(os.path.join(self.download_path, "Flickr_8k.trainImages.txt"), 'r').read().strip().split('\n')))
          test_images = list(set(open(os.path.join(self.download_path, "Flickr_8k.testImages.txt"), 'r').read().strip().split('\n')))

          train_img = [os.path.join(self.download_path, "Flicker8k_Dataset",f) for f in train_images]
          test_img = [os.path.join(self.download_path, "Flicker8k_Dataset",f) for f in test_images]
          self.encoding_train = {}
          self.encoding_test = {}
          for img in train_img:
              img_id = img.split("/")[-1].replace(".jpg","")
              self.encoding_train[img_id] = self.encode(image_features_extract_model,img)[0]
          for img in test_img:
              img_id = img.split("/")[-1].replace(".jpg","")
              self.encoding_test[img_id] = self.encode(image_features_extract_model,img)[0]
          
          with open(os.path.join(self.download_path,"encoded_train_images.pkl"), "wb") as encoded_pickle:
              pickle.dump(self.encoding_train, encoded_pickle)

          with open(os.path.join(self.download_path,"encoded_test_images.pkl"), "wb") as encoded_pickle:
              pickle.dump(self.encoding_test, encoded_pickle)

    def _load(self):
        filename = os.path.join(self.download_path, "Flickr8k.token.txt")
        doc = self.load_doc(filename)
        descriptions = self.load_descriptions(doc)

        self.clean_descriptions(descriptions)
        self.save_descriptions(descriptions, os.path.join(self.download_path,'descriptions.txt'))
        self.vocabulary = self.to_vocabulary(descriptions)
        filename = os.path.join(self.download_path, "Flickr_8k.trainImages.txt")
        train = list(self.load_set(filename))
        filename_test = os.path.join(self.download_path, "Flickr_8k.testImages.txt")
        test = list(self.load_set(filename_test))
        
        self.train_descriptions = self.load_clean_descriptions(os.path.join(self.download_path,'descriptions.txt'), train)
        self.test_descriptions = self.load_clean_descriptions(os.path.join(self.download_path,'descriptions.txt'), test)
        self.descriptions = descriptions
        #print('Descriptions: train=%d' % len(train_descriptions))
        #print(list(train_descriptions)[:10])
        self._load_images_names()
        self._load_vocab()
        #filename = '/content/Flickr_8k.trainImages.txt'
    
    def _load_vocab(self):
      # Create a list of all the training captions
      all_train_captions = []
      for key, val in self.train_descriptions.items():
          for cap in val:
              all_train_captions.append(cap)
      word_count_threshold = 10
      word_counts = {}
      nsents = 0
      for sent in all_train_captions:
          nsents += 1
          for w in sent.split(' '):
              word_counts[w] = word_counts.get(w, 0) + 1

      vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
      print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
      self.vocab_ixtoword = {}
      self.vocab_wordtoix = {}
      ix = 1
      for w in vocab:
          self.vocab_wordtoix[w] = ix
          self.vocab_ixtoword[ix] = w
          ix += 1
      def to_lines(descriptions):
        all_desc = list()
        for key in descriptions.keys():
          [all_desc.append(d) for d in descriptions[key]]
        return all_desc

      # calculate the length of the description with the most words
      def max_length(descriptions):
        lines = to_lines(descriptions)
        return max(len(d.split()) for d in lines)

      # determine the maximum sequence length
      self.vocab_max_length = max_length(self.train_descriptions)
      self.vocab_size  = len(self.vocab_ixtoword) + 1
      #print('Description Length: %d' % max_length)


    def _download_images(self):
        out_file = os.path.join(self.download_path, "Flicker8k_Dataset")
        if not os.path.isdir(out_file):
            out_path = os.path.join(self.download_path, "Flickr8k_Dataset.zip")
            out_path_zip = os.path.join(self.download_path)
            tf.keras.utils.get_file(
                fname=out_path,
                origin="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
            )
            shutil.unpack_archive(out_path, out_path_zip)
            os.remove(out_path)
        else:
            print("imagens existem")

    def _download_text(self):
        out_file = os.path.join(self.download_path, "Flickr8k.token.txt")

        if not os.path.exists(out_file):
            out_path = os.path.join(self.download_path, "Flickr8k_Text.zip")
            out_path_zip = os.path.join(self.download_path)
            tf.keras.utils.get_file(
                fname=out_path,
                origin="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
            )

            shutil.unpack_archive(out_path, out_path_zip)
            os.remove(out_path)
            # print("aqui")

    def _download(self):
        self._download_text()