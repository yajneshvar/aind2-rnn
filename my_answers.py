import numpy as np

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# Fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = np.vstack(tuple([series[i:i+window_size] for i in range(len(series)-window_size)]))
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# Build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    model.add(Dense(1,activation='linear'))
    return model


### Return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    def map_text(c):
       keep = c in punctuation or c in string.ascii_lowercase 
       return  c if keep else  ' '
    clean_text2 = list(map(map_text,text))
    clean_text = [ c for c in text if c in punctuation or c in string.ascii_lowercase]
    text = '';
    for c in clean_text2:
        text += c
    return text

### Fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text) - window_size, step_size):
        tmp_in = text[i:i + window_size]
        tmp_out = text[i + window_size]
        inputs.append(tmp_in)
        outputs.append(tmp_out)

    return inputs,outputs

# Build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200,input_shape=(window_size,num_chars),dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(num_chars))
    model.add(Activation('linear'))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
