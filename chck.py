import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import joblib

# Load saved models
image_model = load_model('tumor_detection_model.h5')
dnn_model = load_model('sms_spam_detection_dnnmodel.h5')
rnn_model = load_model('spam_detection_rnn_model.h5')
perceptron_model = joblib.load('imdb_perceptron_model.pkl')
backprop_model = joblib.load('backprop_model.pkl')
LSTM_model = load_model('imdb_LSTM.h5')

# Streamlit app
st.title("Classification")

# Sidebar
task = st.sidebar.selectbox("Select Task", ["Tumor Detection", "Sentiment Classification"])

def preprocess_message_dnn(message, tokeniser, max_length):
    encoded_message = tokeniser.texts_to_sequences([message])
    padded_message = pad_sequences(encoded_message, maxlen=max_length, padding='post')
    return padded_message

def predict_dnnspam(message, tokeniser, max_length):
    processed_message = preprocess_message_dnn(message, tokeniser, max_length)
    prediction = dnn_model.predict(processed_message)
    return "Spam" if prediction >= 0.5 else "Ham"

# Other prediction functions for sentiment analysis can follow a similar pattern

# Function for CNN prediction
def preprocess_image(image):
    image = image.resize((299, 299))
    image_array = np.array(image)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

def make_prediction_cnn(image, model):
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    preprocessed_image = preprocess_input(img_array)
    prediction = model.predict(preprocessed_image)
    return "Tumor Detected" if prediction > 0.5 else "No Tumor"

if task == "Sentiment Classification":
    st.subheader("Choose Model")
    model_choice = st.radio("Select Model", ["DNN", "RNN", "Perceptron", "Backpropagation", "LSTM"])

    st.subheader("Text Input")
    text_input = st.text_area("Enter Text")

    if st.button("Predict"):
        if model_choice == "DNN":
            # You need to define tokeniser and max_length for DNN model
            prediction_result = predict_dnnspam(text_input, tokeniser, max_length)
            st.write(f"The message is classified as: {prediction_result}")
        # Other model choices should call respective prediction functions similarly

else:
    st.subheader("Choose Model")
    model_choice = st.radio("Select Model", ["CNN"])

    st.subheader("Image Input")
    image_input = st.file_uploader("Choose an image...", type="jpg")

    if image_input is not None:
        image = Image.open(image_input)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            if model_choice == "CNN":
                prediction_result = make_prediction_cnn(image, image_model)
                st.write(prediction_result)
