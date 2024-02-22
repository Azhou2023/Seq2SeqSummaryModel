import numpy as np
import pandas as pd
from util import contraction
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import backend as K
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from util import attention
from matplotlib import pyplot

nltk.download("stopwords")

tf.config.run_functions_eagerly(True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
data = pd.read_csv("./Data/Reviews.csv", nrows=100000)

data.drop_duplicates(subset=["Text"], inplace=True)
data.dropna(axis=0, inplace=True)

#Preprocessing
stop_words = set(stopwords.words('english'))
def clean_text(text):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction.contraction_mapping[t] if t in contraction.contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]

    cleaned_text = []
    for i in tokens:
        if len(i)>=3:
            cleaned_text.append(i)
    return (" ".join(cleaned_text)).strip()

def clean_summary(text):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction.contraction_mapping[t] if t in contraction.contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = newString.split()

    cleaned_text = []
    for i in tokens:
        if len(i)>1:
            cleaned_text.append(i)
    return " ".join(cleaned_text)

cleaned_text = []
for i in data['Text']:
    cleaned_text.append(clean_text(i))
cleaned_summary = []
for i in data['Summary']:
    cleaned_summary.append(clean_summary(i))

data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)
data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x: '_START_' + x + '_END_')

max_len_text=80 
max_len_summary=10

x_tr, x_val, y_tr, y_val = train_test_split(data['cleaned_text'], data['cleaned_summary'], test_size=0.1, random_state=0, shuffle=True);

#Tokenizers
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_tr = pad_sequences(x_tr, maxlen=max_len_text, padding='post')
x_val = x_tokenizer.texts_to_sequences(x_val)
x_val = pad_sequences(x_val, maxlen=max_len_text, padding='post')

x_voc_size = len(x_tokenizer.word_index)+1

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_val = y_tokenizer.texts_to_sequences(y_val)
y_val = pad_sequences(y_val, maxlen=max_len_summary, padding='post')

y_voc_size = len(y_tokenizer.word_index)+1

def build_model():
    K.clear_session()
    latent_dim = 500

    encoder_inputs = Input(shape=(max_len_text))
    enc_emb = Embedding(x_voc_size, latent_dim, trainable=True)(encoder_inputs)

    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    attn_layer = attention.AttentionLayer(name="attention_layer")
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    decoder_concat_input = Concatenate(axis=-1, name="concat_layer")([decoder_outputs, attn_out])

    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

model = build_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
es = EarlyStopping(monitor="val_loss", mode='min', verbose=1)

history = model.fit([x_tr, y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:,1:], epochs=50, callbacks=[es], batch_size=512, validation_data=([x_val, y_val[:,:-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:,1:]))

pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend() 
pyplot.show()



