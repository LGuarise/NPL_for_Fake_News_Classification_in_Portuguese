import numpy as np
import pandas as pd
#import cPickle
#import cv2
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
from keras import layers

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.models import load_model
from keras import regularizers, constraints
from keras.utils import CustomObjectScope
from keras.utils import plot_model

# from glove import Corpus, Glove

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
# import cPickle

EMBEDDING_DIM = int(300)
WORDGRU = int(EMBEDDING_DIM/2)
DROPOUTPER = 0.3

MAX_WORDS = int(3078)
MAX_SENTS = int(410)
MAX_FEATURES = int(3827725)
AVE_WORDS = int(32)
AVE_SENTS = int(21)
WORD_LIMIT = int(100)
SENT_LIMIT = int(427)

latex_special_token = ["!@#$%^&*()"]


class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(AttentionLayer,self).__init__(**kwargs)

	def build(self, input_shape):
		
		#print '\nhi in build attention'
		#print input_shape
	
		self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1], ), name='{}_W'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.bw = self.add_weight(shape=(input_shape[-1], ), name='{}_b'.format(self.name), initializer = 'zero', trainable=True)
		self.uw = self.add_weight(shape=(input_shape[-1], ), name='{}_u'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.trainable_weights = [self.W, self.bw, self.uw]
		
		#print "\nweights in attention"
		#print self.W._keras_shape
		#print self.bw._keras_shape
		#print self.uw._keras_shape
		super(AttentionLayer,self).build(input_shape)
	
	def compute_mask(self, input, mask):
        	return 2*[None]

	def call(self, x, mask=None):
	
		#print '\nhi in attention'
		#print x._keras_shape
		
		uit = K.dot(x, self.W)
		
		#print '\nuit'
		#print uit._keras_shape
		
		uit += self.bw
		uit = K.tanh(uit)

		ait = K.dot(uit, self.uw)
		a = K.exp(ait)

		# apply mask after the exp. will be re-normalized next
		#print mask
		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)

		#print "in att ", K.shape(a)
			
		weighted_input = x * a
		
		#print weighted_input	
		
		ssi = K.sum(weighted_input, axis=1)
		#print "type ", type(ssi)	
		#print "in att si ", theano.tensor.shape(ssi)
		#1111print "hello"
		return [a, ssi]

	def get_output_shape_for(self, input_shape):
		#print '\nhiiiiiiiiiiiiii'
		return  [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]

	def compute_output_shape(self, input_shape):
		#print '\nyooooooooooooooooooooo'
		#print input_shape
		return [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', ';', ':', '•', '–')


def preprocessing1():
    df_train = pd.DataFrame(columns=['classe', 'text'])
    path = '/home/lguarise/Desktop/News_corpus/'

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


def preprocessing2(text, text_t):

    texts = []
    for idx in range(text_t.shape[0]):
        text_x = clean_string(text_t[idx])
        texts.append(text_x)

    text = clean_string(text)
    tokenizer = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
    sentences = tokenizer.tokenize(text)
    tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token=True)
    tokenizer.fit_on_texts(texts)
    data = np.zeros( (SENT_LIMIT, WORD_LIMIT), dtype='int32')
    text_p = []
    for j, sent in enumerate(sentences):
        sent_t = None
        if j < SENT_LIMIT:
            wordTokens = nltk.tokenize.word_tokenize(sent, language='portuguese')
            k = 0
            sent_t = []
            for _, word in enumerate(wordTokens):
                if k < MAX_WORDS and word in tokenizer.word_index and tokenizer.word_index[word] < MAX_FEATURES:
                    data[j, k] = tokenizer.word_index[word]
                    k = k+1
                    sent_t.append(word)
            text_p.append(sent_t)
        else:
            print('sent_limit ultrapassado')
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    return data, word_index, text_p

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
def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	with open(latex_file,'w') as f:
		f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
		string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
		for idx in range(word_num):
			string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
		string += "\n}}}"
		f.write(string+'\n')
		f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


print ('loading files...')

df_train = preprocessing1()
text_t = pd.Series(df_train.text)

with open('text.txt', 'r') as file:
	text = file.read()

data, word_index, text_tokens = preprocessing2(text, text_t)

    
data = np.array(data)
print (data)  
    
#--------------------------------------------------------------------------------------------------------------------------
# MAX_NB_WORDS = len(word_index)+2
# print ("Creating word embedding matrix")
# #TODO: get word2vec pre trained on news corpus
# embeddings_index = {}
# f = open('/home/lguarise/Desktop/skip_s300.txt')
# i = 0
# next(f)
# for line in f:
#     values = line.split()
#     if len(values) == 301:
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
# f.close()
# print('Total words in embedding dictionary: {}'.format(len(embeddings_index)))

# embedding_matrix = np.zeros(shape=(MAX_NB_WORDS, EMBEDDING_DIM), dtype='float32')
# embedding_matrix[1] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word) #glove coeffs wrt to the words
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#     #0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
#     #ref: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py                              
#     else:
#         embedding_matrix[i] = np.random.uniform(-0.25,0.25, EMBEDDING_DIM) 

#-----------------------------------------------------------------------------------------------------------------------------        
#model, model1 = han2(MAX_NB_WORDS, WORD_LIMIT, SENT_LIMIT, EMBEDDING_DIM, WORDGRU, embedding_matrix, DROPOUTPER)
with CustomObjectScope({'AttentionLayer': AttentionLayer}):
	model1 = load_model('2han_model_wordEncoder_epoch_5.h5')
	model = load_model('2han_model_epoch_5.h5')
print (model.summary())
print (model1.summary())
plot_model(model, to_file='model.png')
plot_model(model1, to_file='model1.png')

preds = model.predict([[data]])[0]

print (preds)

embeddings = model1.predict([data])
model_att1  = Model(inputs=model1.input,
				outputs=model1.get_layer('att1').output)

model_att2  = Model(inputs=model.input,
					outputs=model.get_layer('att2').output)

attention1, embeddings_2 = model_att1.predict([data])
attention2, embeddings_3 = model_att2.predict([[data]])

att = []
words = []
for i, sent in enumerate(text_tokens):	
	for j, word in enumerate(sent):
		x = np.sqrt(attention2[0,i,0])*attention1[i,j]
		att.append(x[0]*1000)
		words.append(word)
	color = 'red'
generate(words, att, "sample.tex", color)