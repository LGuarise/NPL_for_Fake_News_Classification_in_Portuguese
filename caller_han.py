import numpy as np
import pandas as pd
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import json

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, model_from_json

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras import regularizers, constraints

from nltk import tokenize
import ast
from keras.preprocessing import sequence

from han import han2
from keras import callbacks

import ast
import sys
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
import gc

CORPUS_PATH = '/home/lguarise/Desktop/News_corpus/'
EMBEDDING_PATH = '/home/lguarise/Desktop/skip_s300.txt'

EMBEDDING_DIM = int(300)
WORDGRU = int(EMBEDDING_DIM/2)
DROPOUTPER = 0.3

MAX_WORDS = int(3078)
MAX_SENTS = int(410)
MAX_FEATURES = int(3827725)
AVE_WORDS = int(32)
AVE_SENTS = int(21)
WORD_LIMIT = int(250)
SENT_LIMIT = int(450)

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', ';', ':', '•', '–')


#---------------------------------------------------------------------------------------------------------------------------
def preprocessing1():
    df_train = pd.DataFrame(columns=['classe', 'text'])
    path = CORPUS_PATH

    for directory in os.listdir(path):
        directory_path = os.path.join(path, directory)
        if os.path.isdir(directory_path):
            count = 0
            if 'fake' in directory:
                TorF = 0
            else:
                TorF = 1
            for filename in os.listdir(directory_path):
                with open(os.path.join(directory_path, filename), 'r') as f:
                    text = f.read()
                    current_df = pd.DataFrame([[TorF, text]],columns=['classe', 'text'])
                    count += 1
                    df_train = df_train.append(current_df, ignore_index=True)
    return df_train


def preprocessing2(text_t,categories,classes):
        """Preprocessing of the text to make it more resonant for training
        """
        paras = []
        labels = []
        texts = []
        for idx in range(text_t.shape[0]):
            text = clean_string(text_t[idx])
            texts.append(text)
            tokenizer = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
            sentences = tokenizer.tokenize(text)
            paras.append(sentences)
        tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token=True)
        tokenizer.fit_on_texts(texts)
        data = np.zeros((len(texts), SENT_LIMIT,
                         WORD_LIMIT), dtype='int32')
        for i, sentences in enumerate(paras):
            for j, sent in enumerate(sentences):
                if j < SENT_LIMIT:
                    wordTokens = nltk.tokenize.word_tokenize(sent, language='portuguese')
                    if len(wordTokens)<6:
                        continue
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k == WORD_LIMIT:
                            print (wordTokens)
                        if k < MAX_WORDS and word in tokenizer.word_index and tokenizer.word_index[word] < MAX_FEATURES:
                            data[i, j, k] = tokenizer.word_index[word]
                            k = k+1
                else:
                    print('sent_limit ultrapassado')
        word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(word_index))
        labels = categories
        print('Shape of data tensor:', data.shape)
        print('Shape of labels tensor:', labels.shape)
        assert (data.shape[0] == labels.shape[0])
        return data, labels, word_index

def clean_string(string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()

def add_dataset(text_t, text, categories, classes, labels):
        try:
            text = pd.concat([text, pd.Series(text_t)])
            categories = pd.concat([categories, pd.Series(labels)])
            assert (len(classes) == categories.unique().tolist())
        except AssertionError:
            print("New class cannot be added in this manner")

def split_dataset(data, labels, validation_split, test_split):
        """
        Spliting the data in train, validation and test
        """
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_test_samples = int(test_split * data.shape[0])
        nb_validation_samples = int(validation_split * data.shape[0])

        x_train = data[:-nb_validation_samples + nb_test_samples]
        y_train = labels[:-nb_validation_samples + nb_test_samples]
        x_val = data[-(nb_validation_samples + nb_test_samples):]
        y_val = labels[-(nb_validation_samples + nb_test_samples):]
        x_test = data[-nb_test_samples:]
        y_test = labels[-nb_test_samples:]
        return x_train, y_train, x_val, y_val, x_test, y_train


print ('loading files...')

df_train = preprocessing1()
text = pd.Series(df_train.text)
categories = pd.Series(df_train.classe)
classes = categories.unique().tolist()
assert (text.shape[0] == categories.shape[0])
data, labels, word_index = preprocessing2(text,categories,classes)
train_articles, train_y, val_articles, val_y, test_articles, test_y = split_dataset(data, labels, 0.2)

del data
gc.collect()
    
train_articles = np.array(train_articles)
train_y = np.array(train_y)

val_articles = np.array(val_articles)
val_y = np.array(val_y)

print ("Train")
print (train_articles.shape)

print ("val")
print (val_articles.shape)    
    
#--------------------------------------------------------------------------------------------------------------------------
MAX_NB_WORDS = len(word_index)+2
#ref:https://richliao.github.io/supervised/classification''/2016/12/26/textclassifier-HATN/
#creating word embeddings using GloVe
print ("Creating word embedding matrix")
#TODO: get word2vec pre trained on news corpus
embeddings_index = {}
f = open(EMBEDDING_PATH)
i = 0
next(f)
for line in f:
    values = line.split()
    if len(values) == 301:
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
print('Total words in embedding dictionary: {}'.format(len(embeddings_index)))
#creating final matrix just for vocabulary words
#all elements in particular the zeroth element is initialized to all zeroes 
#all elements except the zeroth element will be changed later
embedding_matrix = np.zeros(shape=(MAX_NB_WORDS, EMBEDDING_DIM), dtype='float32')
embedding_matrix[1] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word) #glove coeffs wrt to the words
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    #0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones                             
    else:
        embedding_matrix[i] = np.random.uniform(-0.25,0.25, EMBEDDING_DIM) 

#-----------------------------------------------------------------------------------------------------------------------------        
print ('Build model...')
model, model1 = han2(MAX_NB_WORDS, WORD_LIMIT, SENT_LIMIT, EMBEDDING_DIM, WORDGRU, embedding_matrix, DROPOUTPER)
print (model.summary())
print (model1.summary())
#model1 is the word encoder model


print ('Model fit....')


class ModelSave(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        model1.save('SavedModels/2han_model_wordEncoder_epoch_{}.h5'.format(epoch))
        model.save('SavedModels/2han_model_epoch_{}.h5'.format(epoch))
        
modelsave = ModelSave()
        
callbacks = [callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto'),  modelsave]

model.fit(train_articles, train_y, validation_data=(val_articles, val_y), shuffle=True, batch_size=4, epochs=30, callbacks=callbacks)

del train_articles
del train_y
del val_articles
del val_y

'''
test_articles = test_articles[:10]
test_y = test_y[:10]
'''

test_articles = np.array(test_articles)
test_y = np.array(test_y)

print ("test")
print (test_articles.shape)

model.evaluate(test_articles, test_y, batch_size=32)


