from tensorflow.keras.models import load_model       #type:ignore
import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
import pickle
import streamlit as st
path='sentiment_analysis.pkl'
with open(path, 'rb') as file:  
    model = pickle.load(file)
word_index=imdb.get_word_index()
def encode(text):
    token=keras.preprocessing.text.text_to_word_sequence(text)
    token=[word_index[word] if word in word_index else 0 for word in token]
    return sequence.pad_sequences([token],maxlen=250)[0]

def predict(text):
    encoded_text = encode(text)
    pred=np.zeros((1,250))
    pred[0]=encoded_text
    result=model.predict(pred)
    return (f'{result[0][0]:.2%}')

text=st.text_area('enter ur text')
result=predict(text)
button=st.button('predict')
if button:
    st.write(result)



