import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# from gtts import gTTS 
# language = 'en'
# from playsound import playsound

model = ResNet50(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

train_features = load(open("Caption/encoded_train_images.pkl", "rb"))
train_descriptions = load(open("caption/encoded_train_desc.pkl","rb"))

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
#consider only words which occur at least 10 times in the corpus        
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]  

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1      

vocab_size = len(ixtoword) + 1 # one for appended 0's
vocab_size

# convert a dictionary of clean descriptions to a list of descriptions
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
max_length = max_length(train_descriptions)
#print('Description Length: %d' % max_length)

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
# Model saved with Keras model.save()
MODEL_PATH = 'caption/my_model_30'

# Load your trained model
model = load_model(MODEL_PATH)



def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def predict(img):
    #img_y = load_img(img_path,target_size=(224,224))
   # plt.imshow(img_y)
    #plt.show()
    
    img_y1=img_to_array(img)
    img_y1=np.expand_dims(img_y1, axis=0)
    img_y1=preprocess_input(img_y1)
    fea_vec_test = model_new.predict(img_y1) # Get the encoding vector for the image
    fea_vec_test = np.reshape(fea_vec_test, fea_vec_test.shape[1])
    test_reshape = fea_vec_test.reshape(1,2048)
    text = greedySearch(test_reshape)
    #speech = gTTS(text = text, lang = language, slow = False)
    #speech.save('text.mp3')
    
    return text


# with open("Caption/encoded_test_images.pkl", "rb") as encoded_pickle:
#     encoding_test = load(encoded_pickle)

#print("prediction is ",predict('sample_1.jpeg'))
#print("prediction is ",predict('horse.jpg'))
#print("prediction is ",predict('worker.jpg'))
#print("prediction is ",predict('download.png'))
#print("prediction is ",predict('soccer.jpg'))
#print("prediction is ",predict('cat.jpg'))
#print("prediction is ",predict('Ducati.jpg'))
#image = Image.open('photo-1604741637599-b7e53240e1c4.jpg')
#image =image.resize((224,224))
#print("prediction is ",predict(image))





#images = 'Flicker8k_Dataset/'

#z=101
#pic = list(encoding_test.keys())[z]
#image = encoding_test[pic].reshape((1,2048))
#x=plt.imread(images+pic)
#plt.imshow(x)
#plt.show()
#print("Greedy:",greedySearch(image))


