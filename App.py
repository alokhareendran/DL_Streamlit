import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets import imdb

import cv2
from BackPropogation import BackPropogation
from Perceptron import  Perceptron
from sklearn.linear_model import Perceptron
import tensorflow as tf
import joblib
import pickle
from numpy import argmax


# Load saved models
image_model = load_model('tumor_detection_model.h5')
dnn_model = load_model('sms_spam_detection_dnnmodel.h5')
rnn_model = load_model('spam_detection_rnn_model.h5')

# Loading the model using pickle
with open(r'D:/one/OneDrive/Desktop/Streamlit/Model_backprop.pkl', 'rb') as file:
    backprop_model = pickle.load(file)

with open(r'D:/one/OneDrive/Desktop/Streamlit/Percep_model.pkl', 'rb') as file:
    perceptron_model = pickle.load(file)

with open(r'D:/one/OneDrive/Desktop/Streamlit/tokeniser.pkl', 'rb') as file:
    loaded_tokeniser = pickle.load(file)

lstm_model_path='Lstm_model.h5'

# Streamlit app     
st.title("Classification")

# Sidebar
task = st.sidebar.selectbox("Select Task", ["Tumor Detection ", "Sentiment Classification"])
tokeniser = tf.keras.preprocessing.text.Tokenizer()
max_length=10

def predictdnn_spam(text):
        sequence = loaded_tokeniser.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=10)
        prediction = dnn_model.predict(padded_sequence)[0][0]
        if prediction >= 0.5:
            return "not spam"
        else:
            return "spam"
def preprocess_imdbtext(text, maxlen=200, num_words=10000):
    # Tokenizing the text
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text)
    
    # Converting text to sequences
    sequences = tokenizer.texts_to_sequences(text)
    
    # Padding sequences to a fixed length
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    
    return padded_sequences, tokenizer

def predict_sentiment_backprop(text, model):
    preprocessed_text = preprocess_imdbtext(text, 200)  
    prediction = backprop_model.predict(preprocessed_text)
    return prediction

def preprocess_imdb_lstm(user_input, tokenizer, max_review_length=500):
        # Tokenize and pad the user input
        user_input_sequence = tokenizer.texts_to_sequences([user_input])
        user_input_padded = pad_sequences(user_input_sequence, maxlen=max_review_length)
        return user_input_padded

def predict_sentiment_lstm(model, user_input, tokenizer):
        preprocessed_input = preprocess_imdb_lstm(user_input, tokenizer)
        prediction = model.predict(preprocessed_input)
        return prediction

def predict_sentiment_precep(user_input, num_words=1000, max_len=200):
        word_index = imdb.get_word_index()
        input_sequence = [word_index[word] if word in word_index and word_index[word] < num_words else 0 for word in user_input.split()]
        padded_sequence = pad_sequences([input_sequence], maxlen=max_len)
        return padded_sequence 
    
    

def preprocess_message_dnn(message, tokeniser, max_length):
    # Tokenize and pad the input message
    encoded_message = tokeniser.texts_to_sequences([message])
    padded_message = tf.keras.preprocessing.sequence.pad_sequences(encoded_message, maxlen=max_length, padding='post')
    return padded_message

def predict_rnnspam(message, tokeniser, max_length):
    # Preprocess the message
    processed_message = preprocess_message_dnn(message, tokeniser, max_length)
    
    # Predict spam or ham
    prediction = rnn_model.predict(processed_message)
    if prediction >= 0.5:
        return "Spam"
    else:
        return "Ham"


# make a prediction for CNN
def preprocess_image(image):
    image = image.resize((299, 299))
    image_array = np.array(image)
    preprocessed_image = preprocess_input(image_array)

    return preprocessed_image

def make_prediction_cnn(image, image_model):
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))

    preprocessed_image = preprocess_input(img_array)
    prediction = image_model.predict(preprocessed_image)

    if prediction > 0.5:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor")
if task == "Sentiment Classification":
    st.subheader("Choose Model")
    model_choice = st.radio("Select Model", ["DNN", "RNN", "Perceptron", "Backpropagation","LSTM"])

    st.subheader("Text Input")
    

    if model_choice=='DNN':
        text_input = st.text_area("Enter Text")
        if st.button("Predict"):
            if text_input:
                        prediction_result = predictdnn_spam(text_input)
                        st.write(f"The review's class is: {prediction_result}")
            else:
                        st.write("Enter a movie review")

    elif model_choice == "RNN":
            text_input = st.text_area("Enter Text")
            if text_input:
                prediction_result = predict_rnnspam(text_input,loaded_tokeniser,max_length=10)
            if st.button("Predict"):   
                st.write(f"The message is classified as: {prediction_result}")
            else:
                st.write("Please enter some text for prediction")
    elif model_choice == "Perceptron":
                text_input = st.text_area("Enter Text" )
                if st.button('Predict'):
                    processed_input = predict_sentiment_precep(text_input)
                    prediction = perceptron_model.predict(processed_input)[0]
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    st.write(f"Predicted Sentiment: {sentiment}")
    elif model_choice == "LSTM":
            
            lstm_model = tf.keras.models.load_model(lstm_model_path)
            text_input = st.text_area("Enter text for sentiment analysis:", "")
            if st.button("Predict"):
                tokenizer = Tokenizer(num_words=5000)
                prediction = predict_sentiment_lstm(lstm_model, text_input, tokenizer)

                if prediction[0][0]<0.5 :
                    result="Negative"
                    st.write(f"The message is classified as: {result}")
                else:
                    result="Positive"
                    st.write(f"The message is classified as: {result}")

    elif model_choice == "Backpropagation":
                text_input = st.text_area("Enter Text" )
                if st.button('Predict'):
                    processed_input = predict_sentiment_precep(text_input)
                    prediction = backprop_model.predict(processed_input)[0]
                    sentiment = "Positive" if prediction == 1 else "Negative"
                    st.write(f"Predicted Sentiment: {sentiment}")

else:
    st.subheader("Choose Model")
    model_choice = st.radio("Select Model", ["CNN"])

    st.subheader("Image Input")
    image_input = st.file_uploader("Choose an image...", type="jpg")

    if image_input is not None:
        image = Image.open(image_input)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        if st.button("Predict"):
            if model_choice == "CNN":
                make_prediction_cnn(image, image_model)


