import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import string
from string import digits
import re

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model


# %tensorflow_version 2.x
# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))


data= pd.read_table('Notebook\spa.txt',  names =['English', 'Spanish', 'Comments'])
# data = data[:10000]
data.sample(10)

data.English = data.English.apply(lambda word: word.lower())
data.Spanish = data.Spanish.apply(lambda word: word.lower())


data.English = data.English.apply(lambda word: re.sub("'", '', word))
data.Spanish = data.Spanish.apply(lambda word: re.sub("'", '', word))

Punctuations = set(string.punctuation)
data.English = data.English.apply(lambda word: ''.join(char for char in word if char not in Punctuations))
data.Spanish = data.Spanish.apply(lambda word: ''.join(char for char in word if char not in Punctuations))

num_digits = str.maketrans('', '', digits)
data.English = data.English.apply(lambda x: x.translate(num_digits))
data.Spanish = data.Spanish.apply(lambda x: x.translate(num_digits))


data.English = data.English.apply(lambda spaces: spaces.strip())
data.Spanish = data.Spanish.apply(lambda spaces: spaces.strip())

data.English=data.English.apply(lambda x: re.sub(" +", " ", x))
data.Spanish=data.Spanish.apply(lambda x: re.sub(" +", " ", x))


data.Spanish = data.Spanish.apply(lambda sentence : 'START_ '+ sentence + ' _END')
data.sample(10)


English_words = set()
for English in data.English:
    for word in English.split():
        if word not in English_words:
            English_words.add(word)


Spanish_words=set()
for Spanish in data.Spanish:
    for word in Spanish.split():
        if word not in Spanish_words:
            Spanish_words.add(word)


English_words = sorted(list(English_words))
Spanish_words = sorted(list(Spanish_words))


English_length_list=[]
for sentence in data.English:
    English_length_list.append(len(sentence.split(' ')))
max_English_length= max(English_length_list)
print(" Max length of the English sentence",max_English_length)

Spanish_length_list=[]
for sentence in data.Spanish:
    Spanish_length_list.append(len(sentence.split(' ')))
max_Spanish_length= max(Spanish_length_list)
print(" Max length of the Spanish sentence",max_Spanish_length)


English_word2idx = dict([(word, index+1) for index, word in enumerate(English_words)])
Spanish_word2idx = dict([(word, index+1) for index, word in enumerate(Spanish_words)])
vocab_size_Spanish = len(Spanish_word2idx) + 1

English_idx2word = dict([(index, word) for word, index in  English_word2idx.items()])
Spanish_idx2word = dict([(index, word) for word, index in Spanish_word2idx.items()])

# print(English_idx2word, Spanish_idx2word)



#Shuffle the data
data = shuffle(data)



x, y = data.English, data.Spanish
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
x_train.shape, x_test.shape

num_encoder_tokens = len(English_words)
num_decoder_tokens = len(Spanish_words) + 1



def generate_batch(X = x_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for batches in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_English_length),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_Spanish_length),dtype='float32')
            decoder_Spanish_data = np.zeros((batch_size, max_Spanish_length, num_decoder_tokens),dtype='float32')
            for i, (input_text, Spanish_text) in enumerate(zip(X[batches:batches+batch_size], y[batches:batches+batch_size])):
                for t, word in enumerate(input_text.split()):
                  encoder_input_data[i, t] = English_word2idx[word]
                for t, word in enumerate(Spanish_text.split()):
                    if t<len(Spanish_text.split())-1:
                        decoder_input_data[i, t] = Spanish_word2idx[word]
                    if t>0:
                        decoder_Spanish_data[i, t - 1, Spanish_word2idx[word]] = 1.

            yield([encoder_input_data, decoder_input_data], decoder_Spanish_data)

latent_dim = 256 # number of nodes in NN



encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)


encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

encoder_states = [state_h, state_c]



decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

decoder_outputs


# from tensorflow.keras.layers import AdditiveAttention, Concatenate, TimeDistributed

encoder_outputs, decoder_outputs

# attention_result = AdditiveAttention(use_scale=True)([encoder_outputs, decoder_outputs])

# decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)

# decoder_dense = TimeDistributed(Dense(vocab_size_Spanish, activation= 'softmax'))
# decoder_outputs = decoder_dense(decoder_concat_input)



model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

import tensorflow as tf

tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes = True)

train_samples = len(x_train)
val_samples = len(x_test)
batch_size = 128
epochs = 12



# model.fit_generator(generator = generate_batch(x_train, y_train, batch_size = batch_size),
#                     steps_per_epoch = train_samples//batch_size,
#                     epochs=epochs,
#                     validation_data = generate_batch(x_test, y_test, batch_size = batch_size),
#                     validation_steps = val_samples//batch_size)


# model.save_weights('nmt_weights_12epoch.h5')

model.load_weights('Notebook\nmt_weights_12epoch (1).h5')



encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_input)
decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_state_input, [decoder_outputs2] + decoder_states2)



def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)

    Spanish_seq = np.zeros((1,1))

    Spanish_seq[0, 0] = Spanish_word2idx['START_']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([Spanish_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word =Spanish_idx2word[sampled_token_index]
        decoded_sentence += ' '+ sampled_word

        if (sampled_word == '_END' or len(decoded_sentence) > 50):
            stop_condition = True

        Spanish_seq = np.zeros((1,1))
        Spanish_seq[0, 0] = sampled_token_index

        states_value = [h, c]
    return decoded_sentence



train_gen = generate_batch(x_train, y_train, batch_size = 1)
k=-1

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)

print('Input English sentence:', x_train[k:k+1].values[0])
print('Actual Spanish Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Spanish Translation:', decoded_sentence[:-4])



infer_sentence= "how are you"
input_sentence = np.zeros((1, max_English_length),dtype='float32')
for t, word in enumerate(infer_sentence.split()):
  input_sentence[0, t] = English_word2idx[word]
decoded_sentence = decode_sequence(input_sentence)
print(decoded_sentence[:-4])


# !pip install -q gradio

import gradio as gr

def translator(English_sentence):
  input_sentence = np.zeros((1, max_English_length),dtype='float32')
  for t, word in enumerate(English_sentence.split()):
    input_sentence[0, t] = English_word2idx[word]
  decoded_sentence = decode_sequence(input_sentence)
  return decoded_sentence[:-4]

translate_interface = gr.Interface(
    fn = translator,
    inputs = "text",
    outputs = "text",
    title = "English to Spanish Translation"
)

translate_interface.launch(debug=True)

